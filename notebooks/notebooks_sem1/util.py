"""Contains utility function for use in interactive notebooks."""

VARIABLE_NAMES = {'An': '$A_n$', 'Ei': '$E_i$', 'u': '$u$', 'Tlc': '$T_{lc}$', 'Flux': '$F$',
                  'Eabs': '$E_{abs}$', 'gs': '$g_s$', 'E': '$E$', 'gb': '$g_b$', 'psi_head': '$\psi_{h}$', 'FluxC': '$F_C$'}


def _print_scores(train_mean, train_std, cv_mean, cv_std):
    # TODO: use 95th percentile instead of stdev
    print('  - Train: %.4f +/- %.5f' % (train_mean, train_std))
    print('  - CV:    %.4f +/- %.5f' % (cv_mean, cv_std))


def print_cv_scores(cv_scores):
    """Takes the output of sklearn.model_selection.cross_validate
    and prints it out in a readable way"""
    train_mean = cv_scores['train_score'].mean()
    train_std = cv_scores['train_score'].std()
    cv_mean = cv_scores['test_score'].mean()
    cv_std = cv_scores['test_score'].std()
    _print_scores(train_mean, train_std, cv_mean, cv_std)


def print_search_scores(grid_search):
    """Takes a SearchCV model and prints out the fit 
    statistics in in a readable way"""
    train_mean = grid_search.cv_results_[
        'mean_train_score'][grid_search.best_index_]
    train_std = grid_search.cv_results_[
        'std_train_score'][grid_search.best_index_]
    cv_mean = grid_search.cv_results_[
        'mean_test_score'][grid_search.best_index_]
    cv_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    _print_scores(train_mean, train_std, cv_mean, cv_std)


def print_search_best_params(grid_search):
    """Takes a SearchCV model and prints out the optimal
    hyperparameters in a readable way"""
    for k, v in grid_search.best_params_.items():
        print('  - %s = %s' % (k, v))
