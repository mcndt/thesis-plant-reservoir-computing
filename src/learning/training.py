from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, GridSearchCV

from src.learning.scorers import nmse_scorer

from typing import Tuple

ModelScores = Tuple[Tuple[float, float], Tuple[float, float]]


def _print_scores(train_mean, train_std, cv_mean, cv_std):
    print('  - Train: %.4f +/- %.5f' % (train_mean, train_std))
    print('  - CV:    %.4f +/- %.5f' % (cv_mean, cv_std))


def _print_search_best_params(grid_search):
    for k, v in grid_search.best_params_.items():
        print('  - %s = %s' % (k, v))


def get_cv_score(model: BaseEstimator, X, y, groups, folds, verbose=False) -> ModelScores:
    """Performs a cross-validated fit and returns both training 
    and cross-validated scores as a tuple of (mean, std)"""

    cv_scores = cross_validate(model, X, y, groups=groups, cv=folds, scoring=nmse_scorer,
                               return_train_score=True, n_jobs=-1)

    train_mean = cv_scores['train_score'].mean()
    train_std = cv_scores['train_score'].std()
    cv_mean = cv_scores['test_score'].mean()
    cv_std = cv_scores['test_score'].std()

    if verbose:
        print('Cross-validation scores:')
        _print_scores(train_mean, train_std, cv_mean, cv_std)

    return (train_mean, train_std), (cv_mean, cv_std)


def perform_gridsearch(model: BaseEstimator, X, y, groups, folds, param_grid, verbose=False) -> Tuple[GridSearchCV, ModelScores]:
    """Wraps the passed model into GridSearchCV and performs a grid search 
    using the passed parameter grid.

    Returns a tuple of the fitted model wrapped into GridSearchCV 
    and the cross-validated scores after fit. 
    """

    grid_search = GridSearchCV(model, param_grid=param_grid, cv=folds, scoring=nmse_scorer,
                               n_jobs=-1, return_train_score=True)

    grid_search.fit(X, y, groups=groups)

    train_mean = grid_search.cv_results_[
        'mean_train_score'][grid_search.best_index_]
    train_std = grid_search.cv_results_[
        'std_train_score'][grid_search.best_index_]
    cv_mean = grid_search.cv_results_[
        'mean_test_score'][grid_search.best_index_]
    cv_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]

    if verbose:
        print('Cross-validation scores after tuning:')
        _print_scores(train_mean, train_std, cv_mean, cv_std)
        print('Optimal hyperparameters:')
        _print_search_best_params(grid_search)

    return grid_search, ((train_mean, train_std), (cv_mean, cv_std))
