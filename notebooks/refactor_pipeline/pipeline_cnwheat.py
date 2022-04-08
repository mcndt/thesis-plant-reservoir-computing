import sys, os
import numpy as np

from typing import List

sys.path.insert(1, os.path.join(sys.path[0], "../../"))
from src.model.rc_dataset import ExperimentDataset

from pipeline_base import (
    BaseTargetGenerator,
    BaseReservoirGenerator,
    BaseGroupGenerator,
)
from model_config_cnwheat import max_time_step


class TargetGenerator(BaseTargetGenerator):
    """Transforms a dataset or list of datasets into a single target."""

    def __init__(self, *, target: str):
        self.target = target

    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        y = np.empty((1, 0))

        for dataset in datasets:
            assert (
                self.target in dataset.get_targets()
            ), f"{self.target} not available in dataset."
            run_id = dataset.get_run_ids()[0]
            y_dataset = dataset.get_target(self.target, run_id).to_numpy()
            y_dataset = y_dataset[: max_time_step[run_id]]
            y_dataset = y_dataset[24 * warmup_days :]
            y_dataset = y_dataset.reshape((1, -1))
            y = np.concatenate((y, y_dataset), axis=1)

        return y


class SingleReservoirGenerator(BaseReservoirGenerator):
    def __init__(self, *, state_var: str):
        self.state_var = state_var

    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        X = None

        for dataset in datasets:
            assert (
                self.state_var in dataset.get_state_variables()
            ), f"{self.state_var} not available in dataset."
            run_id = dataset.get_run_ids()[0]
            X_dataset = dataset.get_state(self.state_var, run_id)
            X_dataset = X_dataset[: max_time_step[run_id]]
            X_dataset = X_dataset[24 * warmup_days :]

            # Filter out NaN and zero series
            X_NaN = np.isnan(X_dataset)
            NaN_idx = np.any(X_NaN, axis=0)
            X_null = np.isclose(X_dataset, 0)
            null_idx = np.all(X_null, axis=0)
            X_dataset = X_dataset[:, ~NaN_idx & ~null_idx]

            X_dataset = X_dataset.reshape(1, *X_dataset.shape)

            if X is None:
                X = X_dataset
            else:
                X = np.concatenate((X, X_dataset), axis=1)

        return X


class MultiReservoirGenerator(BaseReservoirGenerator):
    def __init__(self, *, state_vars: List[str]):
        self.state_vars = state_vars

    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        X = None
        for state_var in self.state_vars:
            reservoir_generator = SingleReservoirGenerator(state_var=state_var)
            X_raw_var = reservoir_generator.transform(datasets, warmup_days=warmup_days)
            if X is None:
                X = X_raw_var
            else:
                X = np.concatenate((X, X_raw_var), axis=2)
        return X


class TargetReservoirGenerator(BaseReservoirGenerator):
    def __init__(self, *, targets: List[str]):
        self.targets = targets

    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        X = None
        for target in self.targets:
            target_generator = TargetGenerator(target=target)
            X_raw_target = target_generator.transform(datasets, warmup_days=warmup_days)
            X_raw_target = X_raw_target.reshape((*X_raw_target.shape, 1))
            if X is None:
                X = X_raw_target
            else:
                X = np.concatenate((X, X_raw_target), axis=2)
        return X


class GroupGenerator(BaseGroupGenerator):
    def __init__(self, *, day_length: int):
        self.day_length = day_length

    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        groups = np.empty((1, 0))
        for dataset in datasets:
            groups_dataset = self._dataset_group_by_day(dataset, warmup_days)
            groups = np.concatenate((groups, groups_dataset), axis=1)
        return groups

    def _dataset_group_by_day(self, dataset: ExperimentDataset, warmup_days: int):
        run_id = dataset.get_run_ids()[0]
        n_steps = max_time_step[run_id] - self.day_length * warmup_days
        assert (
            n_steps % self.day_length == 0
        ), "Datasets must have a multiple of day length time samples."
        n_groups = n_steps // self.day_length
        groups = np.arange(n_groups).repeat(self.day_length)
        groups = groups.reshape((1, -1))
        return groups

