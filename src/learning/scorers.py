"""Contains functions useful for learning readout functions."""

from sklearn.metrics import mean_squared_error


def nmse_scorer(estimator, X, y_true, epsilon=1e-12):
    """Scores the predictions of given estimator on X against y using
    norm mean squared error.

    Since scikit-learn's default behaviour is to maximize a score function,
    returns negative loss.
    """
    y_pred = estimator.predict(X)
    loss = mean_squared_error(y_true, y_pred) / (y_true.var() + epsilon)
    return -loss
