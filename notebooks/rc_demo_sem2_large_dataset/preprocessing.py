import numpy as np
import pandas as pd


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

