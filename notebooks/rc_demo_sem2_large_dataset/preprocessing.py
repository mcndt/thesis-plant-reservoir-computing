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
    state_size=32,
    warmup_steps=0,
    day_mask=None,
):
    # 1. Take a random subsample of observation nodes
    state_choice = np.random.choice(
        dataset.state_size(), size=state_size, replace=False
    )

    # 2. Cast target and reservoir state into NumPy ndarrays.
    X = np.empty(
        (len(run_ids), dataset.n_steps(), state_size)
    )  # shape (runs, time_steps, nodes)
    y = np.empty((len(run_ids), dataset.n_steps()))  # shape (runs, time_steps)

    for i_run, run_state in enumerate(reservoir_generator):
        X[i_run, :, :] = run_state[:, state_choice]

    for i_run, run_target in enumerate(target_generator):
        y[i_run, :] = run_target

    # 3. Masks are applied.
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

    # 4. Normalize target and reservoir states
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    return X, y

