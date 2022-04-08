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
    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        """Returns shape (sim_runs, time_steps)"""
        pass


class BaseReservoirGenerator(ABC):
    @abstractmethod
    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        pass


class BaseGroupGenerator(ABC):
    def __init__(self, *, day_length: int):
        self.day_length = day_length

    @abstractmethod
    def transform(
        self, datasets: List[ExperimentDataset], *, warmup_days: int
    ) -> np.ndarray:
        pass


##############################
###  Target transformers  ####
##############################


class TargetTransformer(ABC):
    @abstractmethod
    def transform(self, y_raw: np.ndarray) -> np.ndarray:
        pass


class DirectTarget(TargetTransformer):
    def transform(self, y_raw: np.ndarray) -> np.ndarray:
        return y_raw


################################
###  Reservoir transformers  ###
################################


class ReservoirTransformer(ABC):
    @abstractmethod
    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        pass


class DirectReservoir(ReservoirTransformer):
    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        return X_raw


############################
###  Data preprocessing  ###
############################


def flatten(X_tf, y_tf, groups) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert X_tf.shape[:2] == y_tf.shape == groups.shape
    n_runs, n_samples, n_features = X_tf.shape
    X = X_tf.reshape((-1, n_features))
    y = y_tf.reshape((n_runs * n_samples))
    groups = groups.reshape((n_runs * n_samples))
    return X, y, groups


class Preprocessor(ABC):
    @abstractmethod
    def transform(self, X, y, groups) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return X, y, groups


class DaylightMask(Preprocessor):
    def __init__(self, *, day_length: int, start: int, end: int):
        self.daylight_mask = generate_mask(start, end, length=day_length)

    def transform(self, X, y, groups) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert len(X) == len(y) == len(groups)
        n_days = X.shape[0] // len(self.daylight_mask)
        time_mask = np.tile(self.daylight_mask, n_days)
        X = X[time_mask]
        y = y[time_mask]
        groups = groups[time_mask]
        return X, y, groups


class Rescale(Preprocessor):
    def __init__(self, *, per_feature):
        self.per_feature = per_feature

    def transform(self, X, y, groups) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.per_feature:
            X = StandardScaler().fit_transform(X)
        else:
            X = (X - X.mean()) / X.std()
        y = (y - y.mean()) / y.std()
        return X, y, groups


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

    # Data transformation
    warmup_days: int
    target_tf: TargetTransformer
    reservoir_tf: ReservoirTransformer

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


def execute_pipeline(pipeline: RCPipeline):

    # Data generation
    X_raw = pipeline.reservoir.transform(
        pipeline.datasets, warmup_days=pipeline.warmup_days
    )
    y_raw = pipeline.target.transform(
        pipeline.datasets, warmup_days=pipeline.warmup_days
    )
    groups_raw = pipeline.groups.transform(
        pipeline.datasets, warmup_days=pipeline.warmup_days
    )

    # Data transformation
    X_tf = pipeline.reservoir_tf.transform(X_raw)
    y_tf = pipeline.target_tf.transform(y_raw)
    X, y, groups = flatten(X_tf, y_tf, groups_raw)

    # Data processing
    for processor in pipeline.preprocessing:
        X, y, groups = processor.transform(X, y, groups)

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

    return result_dict
