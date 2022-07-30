import sys, os

import numpy as np
import pandas as pd


sys.path.insert(1, os.path.join(sys.path[0], "../../"))
from src.model.rc_dataset import ExperimentDataset


def get_state_random_subset(state: pd.DataFrame, state_size: int) -> pd.DataFrame:
    choice = np.random.choice(state.shape[1], size=state_size, replace=False)
    return state.iloc[:, choice]


def preprocess_data(
    dataset,
    run_ids,
    target_generator,
    reservoir_generator,
    state_size,
    warmup_steps=0,
    day_mask=None,
):
    """
    Preprocessing performed: 

    1. The target signal for each run is computed.
        - Target and reservoir are cast into a ndarray.
    2. Target and reservoir signals are trimmed.
        - A warmup mask is applied to target and reservoir.
        - A night-time mask is applied to target and reservoir.
    3. Target and reservoir are rescaled to zero-mean and unit variance
        - Normalizing transform is fitted on the entire dataset of included experiment runs.
    """

    # 1. Cast target and reservoir state into NumPy ndarrays.
    X = np.empty(
        (len(run_ids), dataset.n_steps(), state_size)
    )  # shape (runs, time_steps, nodes)
    y = np.empty((len(run_ids), dataset.n_steps()))  # shape (runs, time_steps)

    for i_run, run_state in enumerate(reservoir_generator):
        X[i_run, :, :] = run_state

    for i_run, run_target in enumerate(target_generator):
        y[i_run, :] = run_target

    # 2. Masks are applied.
    if day_mask is None:
        time_mask = np.ones(X.shape[1], dtype=bool)
    else:
        n_days = X.shape[1] // len(day_mask)
        assert (
            dataset.n_steps() % len(day_mask) == 0
        ), "Dataset time steps must be multiple of day mask."
        time_mask = np.tile(day_mask, n_days)

    time_mask[:warmup_steps] = False
    X = X[:, time_mask, :]
    y = y[:, time_mask]

    # 3. Normalize target and reservoir states
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    return X, y


def group_by_day(X: np.ndarray, days_per_run, offset_between_runs=1) -> np.ndarray:
    """Simulation state from the same calendar day of simulation inputs, 
    across all runs, are grouped together per day. Shape of X is assumed to be (runs, time_steps, nodes)

    ```
    GROUP 1 | GROUP 2 | GROUP 3 | GROUP 4 | ...
    --------+---------+---------+---------+----
    sim1/d1  sim1/d2   sim1/d3   /         /
    /        sim2/d2   sim2/d3   sim2/d4   /       ...
    /        /         sim3/d3   sim3/d4   sim3/d5 
                                ...                ...
    ```
    """
    n_runs, n_steps, n_nodes = X.shape
    assert (
        n_steps % days_per_run == 0
    ), "steps per run must be an integer multiple of days_per_run."
    steps_per_day = n_steps // days_per_run

    groups = np.empty((n_runs, n_steps))

    offset = 0
    for i_run in range(n_runs):
        groups[i_run, :] = np.arange(offset, offset + days_per_run).repeat(
            steps_per_day
        )
        offset += offset_between_runs

    return groups


def train_test_split_blocks(X, y, groups, test_ratio=0.25, interval_length=16):
    """
    - Train-test splitting is done at group scope (i.e. by calendar day)
    - Training and testing ranges are chosen as contiguous blocks rather 
      than randomly selected.

    e.g. for `interval_length = 8` and `test_ratio = 0.25`, 
    the consecutive groups are assigned as follows:

    ```
    g1     g2      g3      g4      g5      g6      g7     g8                   
    ------+-------+-------+-------+-------+-------+------+------+
    Train | Train | Train | Train | Train | Train | Test | Test | ... (repeat)
    ```
    """
    X_flat = X.reshape((-1, X.shape[-1]))
    y_flat = y.reshape((-1))
    groups_flat = groups.reshape((-1))

    group_ids = np.unique(groups_flat)

    test_run_length = np.ceil(interval_length * test_ratio).astype(int)
    train_run_length = interval_length - test_run_length

    split_mask = np.empty((interval_length), dtype=bool)
    split_mask[0:train_run_length] = True
    split_mask[train_run_length:interval_length] = False
    split_mask = np.tile(
        split_mask, np.ceil(len(group_ids) / interval_length).astype(int)
    )[: len(group_ids)]

    groups_train = group_ids[split_mask]
    groups_test = group_ids[~split_mask]

    train_idx = np.where(np.in1d(groups_flat, groups_train))
    X_train = X_flat[train_idx]
    y_train = y_flat[train_idx]
    groups_train = groups_flat[train_idx]

    test_idx = np.where(np.in1d(groups_flat, groups_test))
    X_test = X_flat[test_idx]
    y_test = y_flat[test_idx]
    groups_test = y_flat[test_idx]

    return (X_train, y_train, groups_train), (X_test, y_test, groups_test)


def direct_target_generator(dataset: ExperimentDataset, target: str, run_ids: [int]):
    """Returns a generator that generates the target from the run id."""
    assert target in dataset.get_targets(), f"{target} not available in dataset."

    # preload data in numpy array for performance reasons
    data = np.empty((len(run_ids), dataset.n_steps()))
    for run_id in run_ids:
        data[run_id, :] = dataset.get_target(target, run_id).to_numpy()

    for run_id in run_ids:
        yield data[run_id, :]


def direct_reservoir_generator(
    dataset: ExperimentDataset,
    state_var: str,
    run_ids: [int],
    state_size=-1,
    random_state=None,
):
    """Returns a function that generates the reservoir from the run id."""
    assert (
        state_var in dataset.get_state_variables()
    ), f"{state_var} not available in dataset."

    if state_size > 0:
        if random_state is not None:
            np.random.seed(random_state)
        state_choice = np.random.choice(
            dataset.state_size(), size=state_size, replace=False
        )
    else:
        state_choice = slice(0, dataset.state_size())

    for run_id in run_ids:
        yield dataset.get_state(state_var, run_id)[:, state_choice]

