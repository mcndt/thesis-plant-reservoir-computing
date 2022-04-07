"""
This file contains functions to be used with RC pipelines for the WheatFspm plant model.
"""
import numpy as np

from model_config import max_time_step, baseline_reservoirs


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
    NaN_idx = np.any(state_NaN, axis=0)

    state_null = np.isclose(state, 0)
    null_idx = np.all(state_null, axis=0)

    state = state[:, ~NaN_idx & ~null_idx]

    yield state


def generate_X_y_groups_baseline(
    *,
    datasets,
    target,
    env_targets,
    prefix,
    target_generator,
    warmup_steps,
    day_mask,
    combined_only=False,
):
    data = {}

    def _preprocess_data(target_data, reservoir_data):
        X_raw, y_raw = preprocess_data(
            target_data, reservoir_data, warmup_steps, day_mask, skip_normalize=True
        )
        X, y = X_raw[0, :, :], y_raw[0, :]
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = (y - y.mean()) / y.std()
        groups = group_by_day(X, day_mask)
        return X, y, groups

    def create_reservoir_from_targets(targets, run_name):
        target_data_list = [
            next(direct_target_generator(dataset, _target, run_name))
            for _target in targets
        ]
        target_data_nd = np.array(target_data_list).T
        return target_data_nd

    # Preprocess the data for each dataset
    for name, dataset in datasets:
        target_data = next(target_generator(dataset, target, name))
        reservoir_data = create_reservoir_from_targets(env_targets, name)
        X, y, groups = _preprocess_data(target_data, reservoir_data)
        data[f"{name}_{prefix}"] = (X, y, groups)

    # Generate the concatenated dataset
    all_arrays = list(data.values())
    X_combined = np.concatenate(list(map(lambda x: x[0], all_arrays)))
    y_combined = np.concatenate(list(map(lambda x: x[1], all_arrays)))
    groups_combined = np.concatenate(list(map(lambda x: x[2], all_arrays)))
    data[f"combined_{prefix}"] = (X_combined, y_combined, groups_combined)
    if combined_only:
        data = {k: v for k, v in data.items() if k.startswith("combined")}

    return data


def generate_X_y_groups(
    *,
    datasets,
    target,
    state_var,
    target_generator,
    state_generator,
    warmup_steps,
    day_mask,
    combined_only=False,
    add_env=False,
):
    """Generates X, y and groups arrays for each dataset, plus a concatenated dataset.
    NOTE: The groups in the concatenated dataset are such that the same calendar day is in the same group.

    Also generates a baseline dataset where the reservoir is just a combination of all environmental inputs.
    """
    data = {}

    def _preprocess_data(target_data, reservoir_data):
        X_raw, y_raw = preprocess_data(
            target_data, reservoir_data, warmup_steps, day_mask
        )
        X, y = X_raw[0, :, :], y_raw[0, :]
        groups = group_by_day(X, day_mask)
        return X, y, groups

    # Preprocess the data for each dataset
    for name, dataset in datasets:
        target_data = next(target_generator(dataset, target, name))
        reservoir_data = next(state_generator(dataset, state_var, name))
        X, y, groups = _preprocess_data(target_data, reservoir_data)
        data[name] = (X, y, groups)

    # Generate the concatenated dataset
    all_arrays = list(data.values())
    X_combined = np.concatenate(list(map(lambda x: x[0], all_arrays)))
    y_combined = np.concatenate(list(map(lambda x: x[1], all_arrays)))
    groups_combined = np.concatenate(list(map(lambda x: x[2], all_arrays)))
    data["combined"] = (X_combined, y_combined, groups_combined)
    if combined_only:
        data = {k: v for k, v in data.items() if k.startswith("combined")}

    # Add environmental baselines
    if add_env:
        for prefix, env_targets in baseline_reservoirs:
            env_data = generate_X_y_groups_baseline(
                datasets=datasets,
                target=target,
                env_targets=env_targets,
                prefix=prefix,
                target_generator=target_generator,
                warmup_steps=warmup_steps,
                day_mask=day_mask,
                combined_only=combined_only,
            )
            data = {**data, **env_data}

    return data


def preprocess_data(
    target, reservoir, warmup_steps=0, day_mask=None, skip_normalize=False
):
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
        # try:
        #     assert (
        #         dataset.n_steps() % len(day_mask) == 0
        #     ), "Dataset time steps must be multiple of day mask."
        # except:
        #     warnings.warn("Dataset time steps is not a multiple of day mask!")
        time_mask = np.tile(day_mask, n_days)

    time_mask[:warmup_steps] = False
    X = X[:, time_mask, :]
    y = y[:, time_mask]

    # 4. Normalize target and reservoir states
    if not skip_normalize:
        X = (X - X.mean()) / X.std()
        y = (y - y.mean()) / y.std()

    return X, y


def preprocess_raw_X(X_raw, warmup_steps=0, day_mask=None):
    X = X_raw
    # 3. Masks are applied.
    if day_mask is None:
        time_mask = np.ones(X.shape[0], dtype=bool)
    else:
        n_days = X.shape[0] // len(day_mask)
        # try:
        #     assert (
        #         dataset.n_steps() % len(day_mask) == 0
        #     ), "Dataset time steps must be multiple of day mask."
        # except:
        #     warnings.warn("Dataset time steps is not a multiple of day mask!")
        time_mask = np.tile(day_mask, n_days)

    time_mask[:warmup_steps] = False
    X = X[: len(time_mask)][time_mask, :]
    # y = y[:, time_mask]

    # 4. Normalize target and reservoir states
    X = (X - X.mean()) / X.std()
    # y = (y - y.mean()) / y.std()

    return X


def group_by_day(X: np.ndarray, day_mask: np.ndarray) -> np.ndarray:
    day_length = day_mask.sum()
    assert len(X) % day_length == 0, "X must have multiple of day_mask time steps."
    n_groups = len(X) // day_length
    groups = np.arange(n_groups).repeat(day_length)
    return groups


def train_test_split_alternating(X, y, groups, ratio=1, blocks=1):
    """The ratio parameter determines how many training days
    are included for every test day.

    e.g. if ratio=2, then the train-test ratio is 2:1
    """
    group_ids = np.unique(groups)

    train_groups_mask = np.ones((ratio + 1), dtype=bool)
    train_groups_mask[-1] = False
    train_groups_mask = np.repeat(train_groups_mask, blocks)
    train_groups_mask = np.tile(
        train_groups_mask, np.ceil(len(group_ids) / len(train_groups_mask)).astype(int)
    )
    train_groups_mask = train_groups_mask[: len(group_ids)]
    train_group_ids = group_ids[train_groups_mask]
    train_mask = np.isin(groups, train_group_ids)

    X_train = X[train_mask]
    groups_train = groups[train_mask]
    y_train = y[train_mask]

    X_test = X[~train_mask]
    y_test = y[~train_mask]
    groups_test = groups[~train_mask]

    return (X_train, y_train, groups_train), (X_test, y_test, groups_test)
