"""Defines a generalized container around RC experiment data using 2D 
tabular data structures for storage as CSV and Pandas based in-memory storage.
"""

import pandas as pd


class Dataset:
    """Represents the data of a single RC experiment with"""

    def __init__(self, csv_path: str):
        self.inputs: pd.DataFrame = None
        self.outputs: pd.DataFrame = None
        self.reservoir: pd.DataFrame = None

        self._load_csv(csv_path)

    def _load_csv(self, csv_path: str):
        pass
