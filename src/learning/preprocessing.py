import numpy as np
import pandas as pd

from typing import Tuple


def generate_mask(h_start, h_end, length=24):
    """Generates a 1D bitmask of size 24 whether to include a sample or not.
    Datum are included if h_start <= time < h_end
    """
    return np.array([h_start <= i < h_end for i in range(length)], dtype=bool)


def preprocess_data(runs, state_var, target, state_size=32,
                    samples_per_run=1, warmup_steps=0, mask=None, mask_length=24) -> Tuple[np.ndarray, np.ndarray]:
    """X, y ndarrays of given data.

    X: ndarray of shape (run, state_sample, day, step, variable)
    y: ndarray of shape (run, state_sample, day, step)
    - The run index means all data came from the same simulation run.
    - The state_sample indicates which leaves were observed.

    state_size (int): the amount of leaves to randomly sample. Default is 32.
    samples_per_run (int): The amount of samples to draw from the total state. Samples will not overlap. Default is 1.
    warmup_steps (int): The amount of time steps to discard as simulation warmup time. Default is 0.

    Assumes that every run has the same state size.
    """
    if runs[0].state_size() < state_size * samples_per_run:
        raise ValueError(
            f'Cannot draw {samples_per_run} samples of size {state_size} from an experiment of total state size {runs[0].state_size()}.')

    if (warmup_steps % 24) != 0:
        raise ValueError(
            f'Only warmup steps of integer days (0, 24, 48, ...) are currently supported.')

    if mask is None:
        mask = np.ones(mask_length, dtype=bool)

    # 1. Aggregate data from each experiment run (and each sample within the run, if samples_per_run > 1)
    data = []  # insert tuples of (state, target, group)

    # choose a random subset of state variables to observe.
    # Assumes the ordering is the same across all runs.
    state_samples = np.random.choice(runs[0].state_size(), size=(
        samples_per_run, state_size), replace=False)

    for i_group, run in enumerate(runs):
        # generate random sample of available leaves to observe
        state_data = run.states[state_var]
        y_i = run.get_target(target)

        for i_sample in range(samples_per_run):
            row_idx = state_samples[i_sample, :]
            X_i = state_data[:, row_idx]
            data.append((X_i, y_i, i_group))

    # 2. Remove sample points using warmup mask and cyclic mask
    for i, (Xi, yi, gi) in enumerate(data):
        run_mask = np.ones(len(Xi), dtype=bool)
        run_mask[:warmup_steps] = False           # warmup mask
        run_mask &= np.tile(mask, len(Xi) // mask_length)  # cyclical mask
        data[i] = Xi[run_mask], yi[run_mask], gi

    # 3. Shape data into ndarray of shape (group, sample, day, step, variable)
    # TODO: make this work with masks that are not an integer amount of days!

    samples_per_day = mask.sum(dtype=int)
    assert(len(data[0][0]) % samples_per_day == 0)
    n_days = len(data[0][0]) // samples_per_day

    X = np.empty((len(runs), samples_per_run, n_days,
                  samples_per_day, state_size))
    y = np.empty((len(runs), samples_per_run, n_days, samples_per_day))

    i_sample, current_group = None, None
    for Xi, yi, gi in data:
        if gi != current_group:
            i_sample, current_group = 0, gi
        Xi = Xi.reshape((n_days, samples_per_day, state_size))
        yi = yi.reshape((n_days, samples_per_day))
        X[current_group, i_sample] = Xi
        y[current_group, i_sample] = yi
        i_sample += 1

    # 4. Normalize X by scaling per variable dimension.
    X_mean = X.mean(axis=(0, 1, 2, 3))
    X_std = X.std(axis=(0, 1, 2, 3))
    X = (X - X_mean) / X_std

    return X, y


def reshape_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reshapes preprocessed data into 2D matrices for model training."""
    n_runs, n_state_samples, n_days, n_steps, n_variables = X.shape
    X = X.reshape((n_runs * n_state_samples * n_days * n_steps, n_variables))
    y = y.reshape((n_runs * n_state_samples * n_days * n_steps))
    return X, y
