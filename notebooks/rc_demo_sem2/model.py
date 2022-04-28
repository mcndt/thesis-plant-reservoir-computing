"""Contains shared libary of model and model utilities for RC plants thesis."""
import sys, os

sys.path.insert(1, os.path.join(sys.path[0], "../../"))
from src.model.rc_dataset import ExperimentDataset


def load_experiment(csv_path: str) -> ExperimentDataset:
    return ExperimentDataset(csv_path=csv_path)


def find_run(iso_date, dataset):
    for i in range(dataset.n_runs()):
        run_index = dataset.get_target("input_Tac", i).index
        if str(run_index[0]).startswith(iso_date):
            return i
    return -1


def scalar_normal_scale_df(df: pd.DataFrame) -> pd.DataFrame:
    array = df.to_numpy()
    mean = array.mean()
    std = array.std()
    array = (array - mean) / std
    return pd.DataFrame(array, columns=df.columns, index=df.index)


def scalar_normal_scale_series(series: pd.Series) -> pd.Series:
    array = series.to_numpy()
    mean = array.mean()
    std = array.std()
    array = (array - mean) / std
    return pd.Series(array, index=series.index)


def split_train_test(array: np.ndarray, day_length: int) -> np.ndarray:
    """Splits the data in a train and test set by alternating days.
  (samples, state_vars) -> (day, day_samples, state_vars)"""
    n_samples = array.shape[0]
    dimensions = array.shape[1:]
    array_days = array.reshape((n_samples // day_length, -1, *array.shape[1:]))
    return array_days[::2], array_days[1::2]


def flatten_data(*array: np.ndarray) -> np.ndarray:
    """Flattens dataset that is grouped per day in the first dimension:
  (n_days, day_length, *dims) -> (n_days * day_length, *dims)"""

    def _flatten_data(array):
        dimensions = array.shape[2:]
        return array.reshape((-1, *dimensions))

    return list(map(_flatten_data, array))


def get_state_random_subset(state: pd.DataFrame, state_size: int) -> pd.DataFrame:
    choice = np.random.choice(state.shape[1], size=state_size, replace=False)
    return state.iloc[:, choice]


def perform_N_fits(
    n_samples: int,
    dataset: ExperimentDataset,
    estimator: BaseEstimator,
    search_grid: dict,
    run_id: int,
    target: str,
    state_var: str,
    state_size: int,
):
    """Randomly samples the state space n_samples times (may overlap between samples).
  Fits a model with Pieters et al. train-test split and grouping strategy, + discarding nighttime data.

  Returns array of test scores.
  """
    state = dataset.get_state(state_var, run_id).sort_index()
    target = target = dataset.get_target(target, run_id)

    # Normalize target and state data to zero mean and unit variance.
    state = scalar_normal_scale_df(state)
    target = scalar_normal_scale_series(target)

    # Apply daylight mask to discard night time samples.
    daylight_mask = generate_mask(5, 21)
    day_length = daylight_mask.sum()
    daylight_mask_run = np.tile(daylight_mask, target.shape[0] // 24)
    state = state.iloc[daylight_mask_run, :]
    target = target.iloc[daylight_mask_run]

    # Reshape target data and generate groups
    y_train, y_test = split_train_test(target.to_numpy(), day_length)

    # Assign CV grouping strategy
    folds = LeaveOneGroupOut()
    groups = np.arange(len(y_train)).repeat(day_length).reshape(y_train.shape)

    # Flatten group and target data
    y_train, y_test, groups = flatten_data(y_train, y_test, groups)

    test_scores = np.zeros((n_samples))

    for i_sample in tqdm(range(n_samples)):
        state_sample = get_state_random_subset(state, state_size)
        X_train, X_test = split_train_test(state_sample.to_numpy(), day_length)
        X_train, X_test = flatten_data(X_train, X_test)
        tuned_readout, tuned_cv_scores = perform_gridsearch(
            readout, X_train, y_train, groups, folds, search_grid, verbose=False
        )
        score = nmse_scorer(tuned_readout, X_test, y_test)
        test_scores[i_sample] = score

    return test_scores

