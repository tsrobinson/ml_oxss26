################################################################################
##                                                                            ##
##                              Oxford Spring School                          ##
##                                Machine Learning                            ##
##                                    Day 1/5                                 ##
##                                                                            ##
##                    Constructing a logistic regression estimator            ##
##                            Walkthrough of extensions                       ##
##                                                                            ##
################################################################################

# These solutions match the exercises in code/1_logistic_sgd.Rmd.
# Run the main notebook first so that the helper functions (sigmoid,
# predict_prob, neg_log_lik) and data objects (X, y, data_all, sgd_fit)
# are already in the environment.

#### Exercise 1. Change the learning rate ####

# Too small: the model converges very slowly (many epochs before the loss
# stabilises). You may hit max_epochs before reaching a good solution.
sgd_slow <- train_logit_sgd(X, y, l_rate = 0.001, max_epochs = 150)
round(sgd_slow$coefficients, 3)
# Compare the final NLL to the main run -- it will be higher (worse) because
# the model has not had enough time to converge.

# Too large: the updates overshoot the optimum. The loss may oscillate wildly
# or even diverge to NaN/Inf.
sgd_fast <- train_logit_sgd(X, y, l_rate = 1.0, max_epochs = 150)
round(sgd_fast$coefficients, 3)
# You will likely see NaN coefficients or very erratic loss values.

# A moderate value works best. The main notebook uses 0.02, which is a good
# compromise for this dataset.


#### Exercise 2. Implement a stopping rule ####

# The idea: once the change in the NLL between consecutive epochs is smaller
# than some tolerance, the model has converged and we can stop early.
# This avoids running unnecessary epochs.

train_logit_sgd_stop <- function(X, y, l_rate = 0.01, max_epochs = 200,
                                  tol = 1e-05) {
  coefficients <- rep(0, ncol(X))
  history <- vector("list", max_epochs)
  previous_nll <- Inf

  for (epoch in seq_len(max_epochs)) {
    for (i in seq_len(nrow(X))) {
      row_vec <- X[i, ]
      yhat_i <- predict_prob(matrix(row_vec, nrow = 1), coefficients)
      gradient <- (yhat_i - y[i]) * row_vec
      coefficients <- coefficients - l_rate * gradient
    }

    train_prob <- predict_prob(X, coefficients)
    epoch_nll <- neg_log_lik(y, train_prob)
    history[[epoch]] <- data.frame(epoch = epoch, nll = epoch_nll)

    message(paste0("Epoch ", epoch, "/", max_epochs,
                   " | NLL = ", round(epoch_nll, 5)))

    if (abs(previous_nll - epoch_nll) < tol) {
      message("Stopping early because the loss has stabilised.")
      break
    }

    previous_nll <- epoch_nll
  }

  list(coefficients = coefficients, history = do.call(rbind, history))
}

sgd_stop <- train_logit_sgd_stop(X, y, l_rate = 0.02, max_epochs = 150,
                                  tol = 1e-05)
round(sgd_stop$coefficients, 3)

# The model should stop well before 150 epochs, saving computation while
# reaching essentially the same coefficients as the full run.

# You could also add a "patience" parameter: only stop after the change has
# been below the tolerance for several consecutive epochs, to guard against
# a single lucky epoch triggering a premature stop.


#### Exercise 3. Add a null predictor X3 ####

set.seed(89)
n <- 500

features_ext <- data.frame(
  X1 = runif(n, -5, 5),
  X2 = runif(n, -2, 2),
  X3 = rnorm(n)       # X3 has no effect on the outcome
)

linear_component <- 3 + 1 * features_ext$X1 - 2 * features_ext$X2 + rnorm(n, 0, 0.15)
p_true <- 1 / (1 + exp(-linear_component))
y_ext <- rbinom(n, 1, p_true)

X_ext <- as.matrix(cbind(Intercept = 1, features_ext))

sgd_ext <- train_logit_sgd(X_ext, y_ext, l_rate = 0.02, max_epochs = 150)
round(sgd_ext$coefficients, 3)

# The coefficient on X3 should be very close to zero, because X3 has no
# relationship with the outcome. The algorithm learns this from the data.
# This is reassuring: SGD does not hallucinate effects that are not there
# (though with small samples it could pick up noise -- try reducing n to 50).


#### Exercise 4 (BONUS). Mini-batch gradient descent ####

train_logit_minibatch <- function(X, y, l_rate = 0.01, max_epochs = 200,
                                   batch_size = 25) {
  coefficients <- rep(0, ncol(X))
  history <- vector("list", max_epochs)
  n <- nrow(X)

  for (epoch in seq_len(max_epochs)) {
    row_order <- sample(seq_len(n))

    # Process in chunks of batch_size
    starts <- seq(1, n, by = batch_size)

    for (s in starts) {
      batch_idx <- row_order[s:min(s + batch_size - 1, n)]
      X_batch <- X[batch_idx, , drop = FALSE]
      y_batch <- y[batch_idx]

      yhat_batch <- predict_prob(X_batch, coefficients)
      # Average gradient over the mini-batch
      gradient <- crossprod(X_batch, yhat_batch - y_batch) / length(batch_idx)
      coefficients <- coefficients - l_rate * drop(gradient)
    }

    train_prob <- predict_prob(X, coefficients)
    epoch_nll <- neg_log_lik(y, train_prob)
    history[[epoch]] <- data.frame(epoch = epoch, nll = epoch_nll)

    message(paste0("Epoch ", epoch, "/", max_epochs,
                   " | NLL = ", round(epoch_nll, 5)))
  }

  list(coefficients = coefficients, history = do.call(rbind, history))
}

sgd_mb <- train_logit_minibatch(X, y, l_rate = 0.02,
                                 max_epochs = 150, batch_size = 25)
round(sgd_mb$coefficients, 3)

# Compare against the row-by-row SGD:
rbind(
  sgd_single = round(sgd_fit$coefficients, 3),
  sgd_minibatch = round(sgd_mb$coefficients, 3)
)

# Mini-batch gradient descent is a middle ground between full-batch GD
# (using all N rows per update) and pure SGD (using 1 row per update).
# Benefits:
# - Smoother gradient estimates than single-row SGD (less noise)
# - Still much faster per epoch than full-batch GD
# - Vectorised matrix operations make each batch step faster in R
# Common batch sizes in practice are 16, 32, 64, or 128.
