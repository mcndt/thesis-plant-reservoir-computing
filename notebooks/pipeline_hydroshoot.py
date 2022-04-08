import sys, os
import numpy as np

from typing import List, Tuple

sys.path.insert(1, os.path.join(sys.path[0], "../../"))
from src.model.rc_dataset import ExperimentDataset

from pipeline_base import (
    BaseTargetGenerator,
    BaseReservoirGenerator,
    BaseGroupGenerator,
    Preprocessor,
)

#########################
###  Data generators  ###
#########################


class TargetGenerator(BaseTargetGenerator):
    """Transforms a dataset or list of datasets into a single target."""

    def __init__(self, *, target: str, run_ids: [int]):
        self.target = target
        self.run_ids = run_ids

    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        assert len(datasets) == 1, "HydroShoot only uses single datasets."
        dataset = datasets[0]
        assert (
            self.target in dataset.get_targets()
        ), f"{self.target} not available in dataset."

        y = None

        for run_id in self.run_ids:
            y_run = dataset.get_target(self.target, run_id).to_numpy()
            y_run = y_run[24 * warmup_days :]
            y_run = y_run.reshape((1, -1))
            if y is None:
                y = y_run
            else:
                y = np.concatenate((y, y_run), axis=0)

        return y


class SingleReservoirGenerator(BaseReservoirGenerator):
    def __init__(
        self, *, state_var: str, run_ids: [int], state_ids: np.ndarray = None,
    ):
        self.state_var = state_var
        self.run_ids = run_ids
        self.state_ids = state_ids

    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        assert len(datasets) == 1, "HydroShoot only uses single datasets."
        dataset = datasets[0]
        assert (
            self.state_var in dataset.get_state_variables()
        ), f"{state_var} not available in dataset."

        if self.state_ids is None:
            self.state_ids = slice(0, dataset.state_size())

        X = None
        for run_id in self.run_ids:
            X_run = dataset.get_state(self.state_var, run_id)[:, self.state_ids]
            X_run = X_run[24 * warmup_days :]

            # Filter out NaN and zero series
            X_NaN = np.isnan(X_run)
            NaN_idx = np.any(X_NaN, axis=0)
            X_run = X_run[:, ~NaN_idx]
            # X_null = np.isclose(X_run, 0)
            # null_idx = np.all(X_null, axis=0)
            # X_run = X_run[:, ~NaN_idx & ~null_idx]

            X_run = X_run[np.newaxis, :, :]

            if X is None:
                X = X_run
            else:
                X = np.concatenate((X, X_run), axis=0)

        return X


class MultiReservoirGenerator(BaseReservoirGenerator):
    def __init__(
        self, *, state_vars: [str], run_ids: [int], state_ids: np.ndarray = None,
    ):
        self.state_vars = state_vars
        self.run_ids = run_ids
        self.state_ids = state_ids

    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        X = None
        for state_var in self.state_vars:
            reservoir_generator = SingleReservoirGenerator(
                state_var=state_var, run_ids=self.run_ids, state_ids=self.state_ids
            )
            X_raw_var = reservoir_generator.transform(datasets, warmup_days=warmup_days)
            if X is None:
                X = X_raw_var
            else:
                X = np.concatenate((X, X_raw_var), axis=2)
        return X


class TargetReservoirGenerator(BaseReservoirGenerator):
    def __init__(self, *, targets: [str], run_ids: [int]):
        self.targets = targets
        self.run_ids = run_ids

    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        X = None
        for target in self.targets:
            target_generator = TargetGenerator(target=target, run_ids=self.run_ids)
            X_target = target_generator.transform(datasets, warmup_days=warmup_days)
            X_target = X_target.reshape((*X_target.shape, 1))
            if X is None:
                X = X_target
            else:
                X = np.concatenate((X, X_target), axis=2)
        return X


class GroupGenerator(BaseGroupGenerator):
    def __init__(self, *, day_length: int, run_ids: [int], days_between_runs: int):
        self.day_length = day_length
        self.run_ids = run_ids
        self.days_between_runs = days_between_runs

    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        assert len(datasets) == 1, "HydroShoot only uses single datasets."
        dataset = datasets[0]
        n_steps = dataset.n_steps()
        assert n_steps % self.day_length == 0, "steps must be multiple of day length"

        n_days = (n_steps // self.day_length) - warmup_days

        groups = np.empty((len(self.run_ids), n_days * self.day_length))
        offset = 0
        for i_run in range(len(self.run_ids)):
            groups[i_run, :] = np.arange(offset, offset + n_days).repeat(
                self.day_length
            )
            offset += self.days_between_runs

        return groups


############################
###  Data preprocessing  ###
############################


class GroupRescale(Preprocessor):
    "Rescale features based on the mean and std of the feature group they belong to."

    def __init__(
        self,
        *,
        datasets: List[ExperimentDataset],
        state_vars: [str],
        state_ids: [int] = None,
    ):
        group_slices, n_features = self._get_groups(datasets, state_vars, state_ids)
        self._group_slices = group_slices
        self._n_expected_features = n_features

    def _get_groups(self, datasets, state_vars, state_ids) -> List[slice]:
        group_slices = []
        offset = 0
        for var in state_vars:
            res_generator = SingleReservoirGenerator(
                state_var=var, run_ids=[0], state_ids=state_ids
            )
            X_raw = res_generator.transform(datasets, warmup_days=0)
            group_size = X_raw.shape[-1]
            group_slices.append(slice(offset, offset + group_size))
            offset += group_size
        return group_slices, offset

    def transform(self, X, y, groups) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for group_idx in self._group_slices:
            X_g = X[:, group_idx]
            X_g = (X_g - X_g.mean()) / (X_g.std() + 1e-12)
            X[:, group_idx] = X_g

        y = (y - y.mean()) / y.std()
        return X, y, groups
