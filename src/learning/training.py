from sklearn.base import BaseEstimator, clone as clone_model
from sklearn.model_selection import cross_validate, GridSearchCV

from src.learning.scorers import nmse_scorer
from src.learning.preprocessing import preprocess_data, reshape_data
from src.learning.grouping import group_by_day

from typing import Tuple, Sequence
from dataclasses import dataclass
from sklearn.model_selection import BaseCrossValidator

import numpy as np

ModelScores = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass
class ModelFitParams:
    """Class definining a model to be fitted."""

    X_name: str
    y_name: str
    dataset_name: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    folds: BaseCrossValidator
    note: str = ""


def _print_scores(train_mean, train_std, cv_mean, cv_std):
    print("  - Train: %.4f +/- %.5f" % (train_mean, train_std))
    print("  - CV:    %.4f +/- %.5f" % (cv_mean, cv_std))


def _print_search_best_params(grid_search):
    for k, v in grid_search.best_params_.items():
        print("  - %s = %s" % (k, v))


def get_cv_score(
    model: BaseEstimator, X, y, groups, folds, verbose=False
) -> ModelScores:
    """Performs a cross-validated fit and returns both training 
    and cross-validated scores as a tuple of (mean, std). 
    Uses NMSE for cross-validation scores."""

    cv_scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=folds,
        scoring=nmse_scorer,
        return_train_score=True,
        n_jobs=-1,
    )

    train_mean = cv_scores["train_score"].mean()
    train_std = cv_scores["train_score"].std()
    cv_mean = cv_scores["test_score"].mean()
    cv_std = cv_scores["test_score"].std()

    if verbose:
        print("Cross-validation scores:")
        _print_scores(train_mean, train_std, cv_mean, cv_std)

    return (train_mean, train_std), (cv_mean, cv_std)


def perform_gridsearch(
    model: BaseEstimator, X, y, groups, folds, param_grid, verbose=False
) -> Tuple[GridSearchCV, ModelScores]:
    """Wraps the passed model into GridSearchCV and performs a grid search 
    using the passed parameter grid.
    Uses NMSE for cross-validation scores.

    Returns a tuple of the fitted model wrapped into GridSearchCV 
    and the cross-validated scores after fit. 
    """

    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=folds,
        scoring=nmse_scorer,
        n_jobs=-1,
        return_train_score=True,
    )

    grid_search.fit(X, y, groups=groups)

    train_mean = grid_search.cv_results_["mean_train_score"][grid_search.best_index_]
    train_std = grid_search.cv_results_["std_train_score"][grid_search.best_index_]
    cv_mean = grid_search.cv_results_["mean_test_score"][grid_search.best_index_]
    cv_std = grid_search.cv_results_["std_test_score"][grid_search.best_index_]

    if verbose:
        print("Cross-validation scores after tuning:")
        _print_scores(train_mean, train_std, cv_mean, cv_std)
        print("Optimal hyperparameters:")
        _print_search_best_params(grid_search)

    return grid_search, ((train_mean, train_std), (cv_mean, cv_std))


def fit_test_model(
    *,
    model: BaseEstimator,
    search_grid: dict,
    scorer,
    train_test_splitter,
    params: ModelFitParams
):
    """TODO: write desc"""
    train, test = train_test_splitter(params.X, params.y, params.groups)

    # search optimal model params
    X_train, y_train, groups_train = train
    cv_model, scores = perform_gridsearch(
        model, X_train, y_train, groups_train, params.folds, search_grid, verbose=False
    )
    (train_mean, train_std), (cv_mean, cv_std) = scores

    # fit with optimal params using all training data
    final_model = clone_model(model)
    final_model.set_params(**cv_model.best_params_)
    final_model.fit(X_train, y_train)

    # Determine test scores
    X_test, y_test, _ = test
    test_score = scorer(final_model, X_test, y_test)
    result_dict = {
        "target": params.y_name,
        "state_var": params.X_name,
        "dataset": params.dataset_name,
        "note": params.note,
        "test_score": test_score,
        "train_mean": train_mean,
        "train_std": train_std,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }

    return (final_model, result_dict)


def evaluate_for_runs(
    model,
    var,
    target,
    runs,
    n_groups,
    folds,
    param_grid,
    verbose=False,
    day_range=None,
    **kwargs
):
    """Trains the readout model for the given target and state variable, using 
    data from the passed runs, using n_group cross-validation.

    Kwargs are passed to the preprocess_data function.

    use day_range kwarg to only fit the model on a subset of the days in the model.
    day_range can be an int (first day of simulation) or tuple (start and end)

    Returns a list of tuples: (target, var, fitted_model, untuned_scores, tuned_scores)"""

    X, y = preprocess_data(runs, var, target, **kwargs)

    if day_range is not None:
        if isinstance(day_range, int):
            start, end = day_range, X.shape[2]
        elif isinstance(day_range, Sequence):
            assert len(day_range) == 2
            start, end = day_range
        X = X[:, :, start:end]
        y = y[:, :, start:end]

    X_train, y_train = reshape_data(X, y)
    groups = group_by_day(X, n_groups=n_groups)

    base_score = get_cv_score(model, X_train, y_train, groups, folds, verbose=verbose)
    tuned_model, tuned_score = perform_gridsearch(
        model, X_train, y_train, groups, folds, param_grid, verbose=verbose
    )

    return (target, var, tuned_model, base_score, tuned_score)
