import numpy as np


def alternating_groups(X: np.ndarray, n_groups=2) -> np.ndarray:
    """Generates a group index array of X where the group ids alternate
    from sample day to sample day (as in Pieters et al. 2021).

    This avoids correlation between adjacent days from introducing bias during training."""
    n_runs, n_state_samples, n_days, n_steps, n_variables = X.shape
    n_data_samples = n_runs * n_state_samples * n_days
    groups = np.empty((n_data_samples), dtype=int)
    for i in range(n_groups):
        groups[i::n_groups] = i
    return groups.repeat(n_steps)


def group_by_day(X: np.ndarray, n_groups=2, offset_between_runs=1) -> np.ndarray:
    """Generates a group index array for X where the same day is 
    always in the same group regardless of simulation run.

    For example, for a (truncated) run lenght of 6 days and 
    an offset of 1, we can create 3 groups as (group id per day):
    groups = [0 1 2 0 1 2 1 2 0 1 2 0 2 0 1 2 0 1 ...] 
    """
    n_runs, n_state_samples, n_days, n_steps, n_variables = X.shape

    if n_state_samples > 1:
        raise NotImplementedError(
            'No implementation for multiple state samples.')

    n_data_samples = n_runs * n_state_samples * n_days
    groups = np.empty((n_data_samples), dtype=int)

    offset = 0
    for i in range(n_runs):
        for j in range(n_days):
            groups[i*n_days + j] = (i*n_days + j + offset) % n_groups
        offset += offset_between_runs

    return groups.repeat(n_steps)
