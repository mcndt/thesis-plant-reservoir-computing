"""
This file contains functions to be used with RC pipelines for the WheatFspm plant model.
"""

import warnings
import numpy as np


def direct_target_generator(dataset, target: str, run_id):
    """Returns a generator that generates the target from the run id."""
    assert target in dataset.get_targets(), f"{target} not available in dataset."
    yield dataset.get_target(target, run_id).to_numpy()[: max_time_step[run_id]]


def direct_reservoir_generator(dataset, state_var: str, run_id):
    """Returns a function that generates the reservoir from the run id."""
    assert (
        state_var in dataset.get_state_variables()
    ), f"{state_var} not available in dataset."

    state = dataset.get_state(state_var, run_id)[: max_time_step[run_id]]

    state_NaN = np.isnan(state)
    state_col_idx = np.all(state_NaN, axis=0)
    state = state[:, ~state_col_idx]

    yield state


def preprocess_data(dataset, target, reservoir, warmup_steps=0, day_mask=None):
    """
    Preprocessing performed:

    1. The target signal for each run is computed.
    2. Target and reservoir are cast into a ndarray.
    3. Target and reservoir signals are trimmed.
      - A warmup mask is applied to target and reservoir.
      - A night-time mask is applied to target and reservoir.
    4. Target and reservoir are rescaled to zero-mean and unit variance
      - Normalizing transform is fitted on the entire dataset of included experiment runs.
    """

    # 1. Cast target and reservoir state into NumPy ndarrays.
    # first dimension is run_id, for compatibility with HydroShoot code
    X = np.empty((1, *reservoir.shape))  # shape (n_runs, time_steps, nodes)
    y = np.empty((1, *target.shape))  # shape (n_runs, time_steps)

    X[0, :, :] = reservoir
    y[0, :] = target

    # 3. Masks are applied.
    if day_mask is None:
        time_mask = np.ones(X.shape[1], dtype=bool)
    else:
        n_days = X.shape[1] // len(day_mask)
        try:
            assert (
                dataset.n_steps() % len(day_mask) == 0
            ), "Dataset time steps must be multiple of day mask."
        except:
            warnings.warn("Dataset time steps is not a multiple of day mask!")
        time_mask = np.tile(day_mask, n_days)

    time_mask[:warmup_steps] = False
    X = X[:, time_mask, :]
    y = y[:, time_mask]

    # 4. Normalize target and reservoir states
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()


def group_by_day(X: np.ndarray, day_mask: np.ndarray) -> np.ndarray:
    day_length = day_mask.sum()
    assert len(X) % day_length == 0, "X must have multiple of day_mask time steps."
    n_groups = len(X) // day_length
    groups = np.arange(n_groups).repeat(day_length)
    return groups


def train_test_split_alternating(X, y, groups, ratio=1):
    """The ratio parameter determines how many training days 
  are included for every test day.
  
  e.g. if ratio=2, then the train-test ratio is 2:1
  """
    group_ids = np.unique(groups)
    test_group_ids = group_ids[ratio :: ratio + 1]
    train_mask = ~np.isin(groups, test_group_ids)

    X_train = X[train_mask]
    groups_train = groups[train_mask]
    y_train = y[train_mask]

    X_test = X[~train_mask]
    y_test = y[~train_mask]
    groups_test = groups[~train_mask]

    return (X_train, y_train, groups_train), (X_test, y_test, groups_test)

