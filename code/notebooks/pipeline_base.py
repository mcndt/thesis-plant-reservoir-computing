import sys, os
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
from dataclasses import dataclass

from sklearn.base import BaseEstimator, clone as clone_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import BaseCrossValidator

sys.path.insert(1, os.path.join(sys.path[0], "../../"))
from src.model.rc_dataset import ExperimentDataset
from src.learning.training import perform_gridsearch
from src.learning.preprocessing import generate_mask


#########################
###  Data generators  ###
#########################


class BaseTargetGenerator(ABC):
    @abstractmethod
    def transform(self, datasets: [ExperimentDataset]) -> np.ndarray:
        """Returns shape (sim_runs, time_steps)"""
        pass


class BaseReservoirGenerator(ABC):
    @abstractmethod
    def transform(self, datasets: [ExperimentDataset]) -> np.ndarray:
        pass


class BaseGroupGenerator(ABC):
    def __init__(self, *, day_length: int):
        self.day_length = day_length

    @abstractmethod
    def transform(self, datasets: [ExperimentDataset]) -> np.ndarray:
        pass


class BaseTimeGenerator(ABC):
    def __init__(self, *, day_length: int):
        self.day_length = day_length

    @abstractmethod
    def transform(self, datasets: [ExperimentDataset]) -> np.ndarray:
        pass


########################################
###  Target/Reservoir transformers  ####
########################################


class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, X, y, groups, time) -> np.ndarray:
        pass


class DirectTransform(BaseTransformer):
    def transform(self, X, y, groups, time) -> np.ndarray:
        return X, y, groups, time


class WarmupTransform(BaseTransformer):
    def __init__(self, *, warmup_days: int, day_length: int):
        self.warmup_steps = warmup_days * day_length

    def transform(self, X, y, groups, time) -> np.ndarray:
        X_tf = X[:, self.warmup_steps :]
        y_tf = y[:, self.warmup_steps :]
        groups_tf = groups[:, self.warmup_steps :]
        time_tf = time[:, self.warmup_steps :]
        return X_tf, y_tf, groups_tf, time_tf


class CustomWarmupTransform(WarmupTransform):
    def __init__(self, *, warmup_steps: int):
        self.warmup_steps = warmup_steps


class DelayLineTransform(BaseTransformer):
    def __init__(self, *, delay_steps: int):
        self.d = delay_steps

    def transform(self, X, y, groups, time) -> np.ndarray:
        if self.d == 0:
            return X, y, groups, time
        elif self.d > 0:
            X_tf = X[:, self.d :]
            y_tf = y[:, : -self.d]
            # Keep the groups tagged to the corresponding reservoir
            groups_tf = groups[:, self.d :]
            time_tf = time[:, self.d :]
            return X_tf, y_tf, groups_tf, time_tf
        else:
            X_tf = X[:, : self.d]
            y_tf = y[:, -self.d :]
            # Keep the groups tagged to the corresponding reservoir
            groups_tf = groups[:, : self.d]
            time_tf = time[:, : self.d]
            return X_tf, y_tf, groups_tf, time_tf


class PolynomialTargetTransform(BaseTransformer):
    def __init__(self, *, poly_coefs: np.ndarray):
        self.coefs = list(reversed(poly_coefs))

    def transform(self, X, y, groups, time) -> np.ndarray:
        y_tf = np.polyval(self.coefs, y)
        # y_tf = y ** self.e
        return X, y_tf, groups, time


class NarmaTargetTransform(BaseTransformer):
    def __init__(self, *, n: int, scale: int, params=(0.3, 0.05, 1.5, 0.1)):
        self.n = n
        self.scale = scale
        self.a, self.b, self.c, self.d = params

    def transform(self, X, y, groups, time) -> np.ndarray:
        _, n_steps = y.shape
        offset = self.n * self.scale

        y_tf = np.zeros(y.shape)
        for t in range(offset - 1, n_steps - 1):
            a_term = self.a * y[:, t]
            b_term = (
                self.b
                * y[:, t]
                * np.sum(y[:, t : t - offset + 1 : -self.scale], axis=1)
            )
            c_term = self.c * y[:, t - offset + 1]
            d_term = self.d
            y_tf[:, t + 1] = a_term + b_term + c_term + d_term

        y_tf = y_tf[:, offset:]
        X_tf = X[:, offset:]
        groups_tf = groups[:, offset:]
        time_tf = time[:, offset:]

        return X_tf, y_tf, groups_tf, time_tf


############################
###  Data preprocessing  ###
############################


def flatten(X_tf, y_tf, groups, time) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert X_tf.shape[:2] == y_tf.shape == groups.shape
    n_runs, n_samples, n_features = X_tf.shape
    X = X_tf.reshape((-1, n_features))
    y = y_tf.reshape((n_runs * n_samples))
    groups = groups.reshape((n_runs * n_samples))
    time = time.reshape((n_runs * n_samples))
    return X, y, groups, time


class Preprocessor(ABC):
    @abstractmethod
    def transform(self, X, y, groups) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return X, y, groups


class DaylightMask(Preprocessor):
    def __init__(self, *, day_length: int, start: int, end: int):
        self.start = start
        self.end = end

    def transform(
        self, X, y, groups, time
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert len(X) == len(y) == len(groups) == len(time)
        daylight_mask = (time >= self.start) & (time < self.end)

        X = X[daylight_mask]
        y = y[daylight_mask]
        groups = groups[daylight_mask]
        time = time[daylight_mask]
        return X, y, groups, time


# class DaylightMask(Preprocessor):
#     def __init__(self, *, day_length: int, start: int, end: int, delay=0):
#         self.daylight_mask = generate_mask(start, end, length=day_length)
#         self.delay = delay

#     def transform(self, X, y, groups) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         assert len(X) == len(y) == len(groups)
#         n_steps = len(X)
#         n_days = int(np.ceil(X.shape[0] / len(self.daylight_mask)))

#         time_mask = np.tile(self.daylight_mask, n_days)
#         if self.delay > 0:
#             time_mask = time_mask[self.delay :]
#         elif self.delay < 0:
#             time_mask = time_mask[: self.delay]

#         X = X[time_mask]
#         y = y[time_mask]
#         groups = groups[time_mask]
#         return X, y, groups


class Rescale(Preprocessor):
    def __init__(self, *, per_feature):
        self.per_feature = per_feature

    def transform(
        self, X, y, groups, time
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.per_feature:
            X = StandardScaler().fit_transform(X)
        else:
            X = (X - X.mean()) / X.std()
        y = (y - y.mean()) / y.std()
        return X, y, groups, time


#######################################
###  Model training and validation  ###
#######################################

sample_set = Tuple[np.ndarray, np.ndarray, np.ndarray]


class BaseTrainTestSplitter(ABC):
    def __init__(self, *, block_size: int, test_ratio: float):
        assert test_ratio <= 0.5, "Test ratio can be at most 50 percent of the data"
        self.block_size = block_size
        self.test_ratio = test_ratio

    @abstractmethod
    def transform(self, X, y, groups) -> Tuple[sample_set, sample_set]:
        return X, y, groups


class TrainTestSplitter(BaseTrainTestSplitter):
    def transform(self, X, y, groups) -> Tuple[sample_set, sample_set]:

        group_ids = np.unique(groups)

        base_length = int(np.round(1 / self.test_ratio))

        train_groups_mask = np.ones((base_length), dtype=bool)
        train_groups_mask[-1] = False
        train_groups_mask = np.repeat(train_groups_mask, self.block_size)
        train_groups_mask = np.tile(
            train_groups_mask,
            np.ceil(len(group_ids) / len(train_groups_mask)).astype(int),
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


#############################
###  Pipeline definition  ###
#############################

BaseScorer = Callable[[BaseEstimator, np.ndarray, np.ndarray], float]


@dataclass
class RCPipeline:
    # Metadata (added as columns in the results dict)
    metadata: dict

    # Data generation
    datasets: List[ExperimentDataset]
    target: BaseTargetGenerator
    reservoir: BaseReservoirGenerator
    groups: BaseGroupGenerator
    time: BaseTimeGenerator

    # Data transformation
    transforms: List[BaseTransformer]

    # Data preproccessing
    preprocessing: List[Preprocessor]

    # Model
    train_test_split: BaseTrainTestSplitter
    readout_model: BaseEstimator
    model_param_grid: dict
    model_scorer: BaseScorer
    folds: BaseCrossValidator


############################
###  Pipeline execution  ###
############################


def fit_model(model, X, y, groups, folds, search_grid) -> BaseEstimator:
    """Optimizes model using param search grid and returns the optimal model."""
    cv_model, cv_scores = perform_gridsearch(
        model, X, y, groups, folds, search_grid, verbose=False
    )
    final_model = clone_model(model)
    final_model.set_params(**cv_model.best_params_)
    final_model.fit(X, y)
    return final_model, cv_scores


def execute_pipeline(pipeline: RCPipeline, return_model_data=False):

    # Data generation
    X_raw = pipeline.reservoir.transform(pipeline.datasets)
    y_raw = pipeline.target.transform(pipeline.datasets)
    groups_raw = pipeline.groups.transform(pipeline.datasets)
    time_raw = pipeline.time.transform(pipeline.datasets)

    # Data transformation
    X_tf, y_tf, groups_tf, time_tf = X_raw, y_raw, groups_raw, time_raw
    for transform in pipeline.transforms:
        X_tf, y_tf, groups_tf, time_tf = transform.transform(
            X_tf, y_tf, groups_tf, time_tf
        )

    X, y, groups, time = flatten(X_tf, y_tf, groups_tf, time_tf)

    # Data processing
    for processor in pipeline.preprocessing:
        try:
            X, y, groups, time = processor.transform(X, y, groups, time)
        except Exception as e:
            print(f"Error in processor: {processor.__class__}")
            raise e

    # Train test splitting
    train, test = pipeline.train_test_split.transform(X, y, groups)

    # Model training
    X_train, y_train, groups_train = train
    final_model, cv_scores = fit_model(
        pipeline.readout_model,
        X_train,
        y_train,
        groups_train,
        pipeline.folds,
        pipeline.model_param_grid,
    )
    (train_mean, train_std), (cv_mean, cv_std) = cv_scores

    # Model evaluation
    X_test, y_test, _ = test
    test_score = pipeline.model_scorer(final_model, X_test, y_test)
    result_dict = {
        **pipeline.metadata,
        "test_score": test_score,
        "train_mean": train_mean,
        "train_std": train_std,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
    }

    if return_model_data is True:
        model_data = {
            "train_data": (X_train, y_train),
            "test_data": (X_test, y_test),
            "final_model": final_model,
        }
        return (result_dict, model_data)

    return result_dict
