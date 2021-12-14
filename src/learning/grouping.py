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

