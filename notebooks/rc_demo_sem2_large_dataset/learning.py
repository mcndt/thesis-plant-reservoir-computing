import numpy as np


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

