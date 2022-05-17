import sys, os
import numpy as np

from typing import List, Tuple

sys.path.insert(1, os.path.join(sys.path[0], "../../"))
from src.model.rc_dataset import ExperimentDataset

from pipeline_base import (
    BaseTargetGenerator,
    BaseReservoirGenerator,
    BaseGroupGenerator,
    BaseTimeGenerator,
    Preprocessor,
)
from model_config_cnwheat import max_time_step


#########################
###  Data generators  ###
#########################


class TargetGenerator(BaseTargetGenerator):
    """Transforms a dataset or list of datasets into a single target."""

    def __init__(self, *, target: str):
        self.target = target

    def transform(self, datasets: List[ExperimentDataset]) -> np.ndarray:

        if self.target == "output__custom__PARa":
            generator = AbsorbedPARGenerator()
            return generator.transform(datasets)

        y = np.empty((1, 0))

        for dataset in datasets:
            assert (
                self.target in dataset.get_targets()
            ), f"{self.target} not available in dataset."
            run_id = dataset.get_run_ids()[0]
            y_dataset = dataset.get_target(self.target, run_id).to_numpy()
            y_dataset = y_dataset[: max_time_step[run_id]]
            y_dataset = y_dataset.reshape((1, -1))
            y = np.concatenate((y, y_dataset), axis=1)

        return y


class AbsorbedPARGenerator(BaseTargetGenerator):
    """Transforms a dataset or list of datasets into a single target."""

    def transform(self, datasets: List[ExperimentDataset]) -> np.ndarray:
        reservoir = SingleReservoirGenerator(state_var="state__PARa")
        data = reservoir.transform(datasets)
        data = np.sum(data, axis=-1)
        return data


class SingleReservoirGenerator(BaseReservoirGenerator):
    def __init__(
        self, *, state_var: str, state_ids: np.ndarray = None,
    ):
        self.state_var = state_var
        self.state_ids = state_ids

    def transform(self, datasets: List[ExperimentDataset]) -> np.ndarray:
        X = None

        for dataset in datasets:
            assert (
                self.state_var in dataset.get_state_variables()
            ), f"{self.state_var} not available in dataset."
            run_id = dataset.get_run_ids()[0]
            X_dataset = dataset.get_state(self.state_var, run_id)
            X_dataset = X_dataset[: max_time_step[run_id]]

            # Filter out NaN and zero series
            X_NaN = np.isnan(X_dataset)
            NaN_idx = np.any(X_NaN, axis=0)
            X_null = np.isclose(X_dataset, 0)
            null_idx = np.all(X_null, axis=0)
            X_dataset = X_dataset[:, ~NaN_idx & ~null_idx]

            if self.state_ids is not None:
                X_dataset = X_dataset[:, self.state_ids]

            X_dataset = X_dataset.reshape(1, *X_dataset.shape)

            if X is None:
                X = X_dataset
            else:
                X = np.concatenate((X, X_dataset), axis=1)

        return X


class MultiReservoirGenerator(BaseReservoirGenerator):
    def __init__(
        self, *, state_vars: List[str], state_ids: np.ndarray = None,
    ):
        self.state_vars = state_vars
        self.state_ids = state_ids

    def transform(self, datasets: List[ExperimentDataset]) -> np.ndarray:
        X = None
        for state_var in self.state_vars:
            reservoir_generator = SingleReservoirGenerator(
                state_var=state_var, state_ids=self.state_ids
            )
            X_raw_var = reservoir_generator.transform(datasets)
            if X is None:
                X = X_raw_var
            else:
                X = np.concatenate((X, X_raw_var), axis=2)
        return X


class TargetReservoirGenerator(BaseReservoirGenerator):
    def __init__(self, *, targets: List[str]):
        self.targets = targets

    def transform(self, datasets: List[ExperimentDataset]) -> np.ndarray:
        X = None
        for target in self.targets:
            target_generator = TargetGenerator(target=target)
            X_raw_target = target_generator.transform(datasets)
            X_raw_target = X_raw_target.reshape((*X_raw_target.shape, 1))
            if X is None:
                X = X_raw_target
            else:
                X = np.concatenate((X, X_raw_target), axis=2)
        return X


class GroupGenerator(BaseGroupGenerator):
    def __init__(self, *, day_length: int):
        self.day_length = day_length

    def transform(self, datasets: List[ExperimentDataset]) -> np.ndarray:
        groups = np.empty((1, 0))
        for dataset in datasets:
            groups_dataset = self._dataset_group_by_day(dataset)
            groups = np.concatenate((groups, groups_dataset), axis=1)
        return groups

    def _dataset_group_by_day(self, dataset: ExperimentDataset):
        run_id = dataset.get_run_ids()[0]
        n_steps = max_time_step[run_id]
        assert (
            n_steps % self.day_length == 0
        ), "Datasets must have a multiple of day length time samples."
        n_groups = n_steps // self.day_length
        groups = np.arange(n_groups).repeat(self.day_length)
        groups = groups.reshape((1, -1))
        return groups


class TimeGenerator(BaseTimeGenerator):
    def __init__(self, *, day_length: int):
        self.day_length = day_length

    def transform(self, datasets: List[ExperimentDataset]) -> np.ndarray:
        time = np.empty((1, 0))
        for dataset in datasets:
            time_dataset = self._dataset_transform(dataset)
            time = np.concatenate((time, time_dataset), axis=1)
        return time

    def _dataset_transform(self, dataset) -> np.ndarray:
        run_id = dataset.get_run_ids()[0]
        n_steps = max_time_step[run_id]
        n_days = int(np.ceil(n_steps / self.day_length))

        mask = np.arange(self.day_length)
        mask = np.tile(mask, n_days)
        mask = mask[np.newaxis, :n_steps]
        return mask


############################
###  Data preprocessing  ###
############################


class GroupRescale(Preprocessor):
    "Rescale features based on the mean and std of the feature group they belong to."

    def __init__(
        self,
        *,
        datasets: List[ExperimentDataset],
        state_vars: List[str],
        state_ids: [int] = None,
    ):
        group_slices, n_features = self._get_groups(datasets, state_vars, state_ids)
        self._group_slices = group_slices
        self._n_expected_features = n_features

    def _get_groups(self, datasets, state_vars, state_ids,) -> List[slice]:
        group_slices = []
        offset = 0
        for var in state_vars:
            res_generator = SingleReservoirGenerator(state_var=var, state_ids=state_ids)
            X_raw = res_generator.transform(datasets)
            group_size = X_raw.shape[-1]
            group_slices.append(slice(offset, offset + group_size))
            offset += group_size
        return group_slices, offset

    def transform(
        self, X, y, groups, time
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        for group_idx in self._group_slices:
            X_g = X[:, group_idx]
            X_g = (X_g - X_g.mean()) / X_g.std()
            X[:, group_idx] = X_g

        y = (y - y.mean()) / y.std()
        return X, y, groups, time
