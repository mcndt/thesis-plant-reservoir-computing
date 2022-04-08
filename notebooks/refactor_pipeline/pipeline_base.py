import sys, os
import numpy as np

from abc import ABC, abstractmethod
from typing import List

sys.path.insert(1, os.path.join(sys.path[0], "../../"))
from src.model.rc_dataset import ExperimentDataset


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

