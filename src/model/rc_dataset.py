import pandas as pd
import numpy as np


class ExperimentDataset:
    """Wrapper class for handling a dataset from a single RC experiment run."""

    def __init__(self, path: str):
        """Load a experiment dataset from csv file."""
        self._inputs: pd.DataFrame = None
        self._outputs: pd.DataFrame = None
        self._state: pd.DataFrame = None
        self.load_data(path)

    def load_data(self, path: str):
        dataset_df = pd.read_csv(csv_path)
        self._inputs = dataset_df[dataset_df["type"] == "INPUT"].dropna(
            how="all", axis=1
        )
        self._outputs = dataset_df[dataset_df["type"] == "OUTPUT"].dropna(
            how="all", axis=1
        )
        self._state = dataset_df[dataset_df["type"] == "STATE"].dropna(
            how="all", axis=1
        )
        self._state["state_id"] = self._state["state_id"].astype(int)
        assert len(self._inputs) == len(
            self._outputs
        ), "Input and output set have different lengths."

    def get_input_variables(self) -> tuple:
        """Get the input keys available."""
        input_col_names = self._inputs.columns
        return tuple(filter(lambda x: x.startswith("input_"), input_col_names))

    def get_output_variables(self) -> tuple:
        """Get the input output available."""
        output_col_names = self._outputs.columns
        return tuple(filter(lambda x: x.startswith("output_"), output_col_names))

    def get_targets(self) -> tuple:
        """Get the target keys available."""
        return (*self.get_input_variables(), *self.get_output_variables())

    def get_state_variables(self) -> tuple:
        """Get the state variables available."""
        state_col_names = self._state.loc[
            :, ~self._state.columns.isin(["state_id", "state_type"])
        ].columns
        return tuple(filter(lambda x: x.startswith("state_"), state_col_names))

    def n_steps(self) -> int:
        return len(self._inputs)

    def state_size(self) -> int:
        return len(self._state.groupby("state_id"))

    def get_target(self, target_key) -> pd.Series:
        "Get a target signal as pandas Series by the target key."
        assert (
            target_key in self.get_targets()
        ), f"{target_key} not in available targets."
        source = self._inputs if target_key.startswith("input_") else self._outputs
        target_series = source[target_key]
        target_series.index = source["time"]
        return target_series

    def get_state(self, state_key) -> pd.DataFrame:
        "Get the entire reservoir state of variable as pandas DataFrame by the state key."
        assert (
            state_key in self.get_state_variables()
        ), f"{state_key} not in available state variables."
        return self._state.pivot(index="time", columns=["state_id"], values=state_key)

