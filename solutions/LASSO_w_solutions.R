################################################################################
##                                                                            ##
##                              Oxford Spring School                          ##
##                                Machine Learning                            ##
##                                    Day 2/5                                 ##
##                                                                            ##
##                          LASSO Models and Post-Double Selection            ##
##                            Walkthrough of extensions                       ##
##                                                                            ##
################################################################################

# These solutions match the exercises in code/2_LASSO.Rmd.
# Run the main notebook first so that the data objects (emw, X, V, VX, D, DV,
# Y, RHS_matrix, controls, mod_mat) and functions (cv_lasso_selector) are
# already in the environment.

library(haven)
library(glmnet)


#### Exercise 1. lambda.min vs lambda.1se ####

# The main notebook uses lambda.1se (the most regularised model whose CV error
# is within one standard error of the minimum). Switching to lambda.min picks
# the lambda that minimises CV error directly, which is less conservative.

set.seed(89)

Y_lasso_min <- cv.glmnet(x = RHS_matrix, y = Y, alpha = 1,
                          family = "binomial", nfolds = 10)
D_lasso_min <- cv.glmnet(x = RHS_matrix, y = D, alpha = 1,
                          family = "gaussian", nfolds = 10)
DV_lasso_min <- cv.glmnet(x = RHS_matrix, y = DV, alpha = 1,
                           family = "gaussian", nfolds = 10)

count_nonzero <- function(fit, s) {
  coefs <- as.vector(coef(fit, s = s))[-1]  # drop intercept
  sum(coefs != 0)
}

cat("Y  — lambda.1se nonzero:", count_nonzero(Y_lasso_min, "lambda.1se"),
    "| lambda.min nonzero:", count_nonzero(Y_lasso_min, "lambda.min"), "\n")
cat("D  — lambda.1se nonzero:", count_nonzero(D_lasso_min, "lambda.1se"),
    "| lambda.min nonzero:", count_nonzero(D_lasso_min, "lambda.min"), "\n")
cat("DV — lambda.1se nonzero:", count_nonzero(DV_lasso_min, "lambda.1se"),
    "| lambda.min nonzero:", count_nonzero(DV_lasso_min, "lambda.min"), "\n")

# lambda.min selects MORE variables because it applies less regularisation.
# The trade-off: lambda.min may overfit, while lambda.1se may be too aggressive
# in dropping variables. For variable selection (Stage 1 of PDS) the more
# conservative lambda.1se is usually preferred to avoid including noise.

# You can see this visually:
plot(Y_lasso_min, main = "CV curve for Y — note the two vertical lines")


#### Exercise 2. Simulate sparse data and test LASSO recovery ####

set.seed(89)
n_sim <- 2000
p_sim <- 100

X_sim <- matrix(runif(n_sim * p_sim, -5, 5), nrow = n_sim, ncol = p_sim)
colnames(X_sim) <- paste0("X", 1:p_sim)

# True coefficients: X1 = 10, X2 = -5, X3-X10 = 0, X11-X100 ~ N(0, 0.05)
beta_true <- c(10, -5, rep(0, 8), rnorm(90, 0, 0.05))
Y_sim <- X_sim %*% beta_true + rnorm(n_sim, 0, 1)

cv_sim <- cv.glmnet(x = X_sim, y = Y_sim, alpha = 1, nfolds = 10)
coef_sim <- as.vector(coef(cv_sim, s = "lambda.1se"))[-1]

# Inspect the key coefficients:
cat("\nTrue vs estimated coefficients for the strong signals:\n")
data.frame(
  variable = c("X1", "X2"),
  true = c(10, -5),
  lasso = round(coef_sim[1:2], 3)
)

cat("\nX3-X10 (true zero) — estimated:\n")
round(coef_sim[3:10], 4)

cat("\nNon-zero among X11-X100:", sum(coef_sim[11:100] != 0), "out of 90\n")

# Key observations:
# 1. X1 and X2 are recovered with large coefficients (slightly shrunk)
# 2. X3-X10 are almost always set to exactly zero — good!
# 3. Some of X11-X100 may survive if their true effect happened to be
#    larger than the shrinkage threshold. Most are correctly zeroed out.


#### Exercise 3. Add a proper train/test split ####

set.seed(42)

# Hold out 20% as a genuine test set
n_total <- nrow(emw)
test_idx <- sample(seq_len(n_total), size = floor(0.2 * n_total))
train_idx <- setdiff(seq_len(n_total), test_idx)

# Rebuild the matrices on training data only
X_tr <- X[train_idx, ]
V_tr <- V[train_idx]
VX_tr <- VX[train_idx, ]
D_tr <- D[train_idx]
DV_tr <- DV[train_idx]
Y_tr <- Y[train_idx]

RHS_tr <- RHS_matrix[train_idx, ]

# Stage 1: LASSO selection on training data only
Y_lasso_tr <- cv_lasso_selector(LHS = Y_tr, RHS = RHS_tr)
D_lasso_tr <- cv_lasso_selector(LHS = D_tr, RHS = RHS_tr)
DV_lasso_tr <- cv_lasso_selector(LHS = DV_tr, RHS = RHS_tr)

sel_cols <- sort(unique(c(Y_lasso_tr$index, D_lasso_tr$index, DV_lasso_tr$index)))
sel_names <- colnames(RHS_tr)[sel_cols]

cat("Selected variables (train only):", length(sel_cols), "\n")

# Stage 2: inference model on training data
sel_ints <- setdiff(sel_names, c("V", colnames(X)))

ds_train <- data.frame(
  Protest = Y_tr,
  remit = D_tr,
  dict = V_tr,
  remit_dict = DV_tr,
  as.data.frame(X_tr),
  as.data.frame(RHS_tr[, sel_ints, drop = FALSE])
)
ds_train <- ds_train[, !duplicated(names(ds_train))]

ds_model_tr <- lm(Protest ~ ., data = ds_train)

# Naive model on training data
naive_model_tr <- lm(
  Protest ~ remit * dict + l1gdp + l1pop + l1nbr5 + l12gr + l1migr + elec3 +
    factor(period) + factor(cowcode),
  data = emw[train_idx, ]
)

# Evaluate on genuinely held-out test data
X_te <- X[test_idx, ]
RHS_te <- RHS_matrix[test_idx, ]

ds_test <- data.frame(
  Protest = Y[test_idx],
  remit = D[test_idx],
  dict = V[test_idx],
  remit_dict = DV[test_idx],
  as.data.frame(X_te),
  as.data.frame(RHS_te[, sel_ints, drop = FALSE])
)
ds_test <- ds_test[, !duplicated(names(ds_test))]

mse <- function(y, yhat) mean((y - yhat)^2)

naive_test_p <- predict(naive_model_tr, newdata = emw[test_idx, ])
pds_test_p <- predict(ds_model_tr, newdata = ds_test)

cat("\n--- Held-out test performance ---\n")
cat("Naive model  MSE:", round(mse(Y[test_idx], naive_test_p), 4), "\n")
cat("PDS model    MSE:", round(mse(Y[test_idx], pds_test_p), 4), "\n")

# Because the test set is now genuinely unseen, these numbers are an honest
# estimate of out-of-sample performance. Compare to the numbers in the main
# notebook where both models were fit on the full data — those were optimistic.


#### Exercise 4. Reading guide for Blackwell and Olson ####

# The full paper is:
#   Blackwell, Matthew and Michael P. Olson. "Reducing Model
#   Misspecification and Bias in the Estimation of Interactions."
#   Political Analysis, 2021.
#   https://doi.org/10.1017/pan.2021.19

# Key simplifications in our exercise compared to the full paper:
#
# 1. Penalty loadings: The paper uses observation-specific penalty factors
#    (penalty.factor argument in glmnet) that are iteratively refined.
#    Our exercise uses the default uniform penalty.
#
# 2. Clustered standard errors: The paper accounts for clustering (e.g.,
#    by country) in both the LASSO penalty calibration and the Stage 2
#    inference. We ignore clustering entirely.
#
# 3. Lambda calibration: The paper derives a theoretically motivated lambda
#    rather than relying purely on cross-validation. See their rlasso_cluster
#    function for details.
#
# 4. Complete-case analysis: We use na.omit as a shortcut. The paper's
#    application handles missing data more carefully.
#
# 5. The paper's method is designed for conjoint experiments where the
#    treatment is randomly assigned. Our application to observational data
#    relies on stronger assumptions about selection on observables.
