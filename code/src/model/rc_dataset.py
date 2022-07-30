import pandas as pd
import numpy as np


class ExperimentDataset:
    """Wrapper class for handling a dataset from a single RC experiment run."""

    def __init__(self, dataset_df=None, csv_path=None):
        """Load a experiment dataset from csv file."""
        self._inputs: pd.DataFrame = None
        self._outputs: pd.DataFrame = None
        self._state: pd.DataFrame = None

        # Stores the names of the input targets
        self._input_keys = None
        # Store the names of the output targets
        self._output_keys = None
        # Stores the names of the available state variables.
        self._state_keys = None
        # Stores the names of the available experiment runs.
        self._run_keys = None
        # Stores the names of the available state nodes.
        # NOTE: Not every node necessarily contains every state variable
        #       (e.g. one node can be a leaf while the other is a stem).
        self.state_keys = None

        self._run_map = None
        self._node_map = None
        self._state_nd = None

        if dataset_df is not None:
            self.load_data(dataset_df)
        elif csv_path is not None:
            self.load_dataframe(csv_path)
        else:
            raise Exception('Must set kwarg "dataset_df" or "csv_path"')

    def load_dataframe(self, csv_path):
        df = pd.read_csv(csv_path)
        self.load_data(df)

    def load_data(self, dataset_df):
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

        input_col_names = self._inputs.columns
        self._input_keys = tuple(
            filter(lambda x: x.startswith("input_"), input_col_names)
        )
        output_col_names = self._outputs.columns
        self._output_keys = tuple(
            filter(lambda x: x.startswith("output_"), output_col_names)
        )
        state_col_names = self._state.loc[
            :, ~self._state.columns.isin(["state_id", "state_type"])
        ].columns
        self._state_keys = tuple(
            filter(lambda x: x.startswith("state_"), state_col_names)
        )

        self._run_keys = self._inputs["run_id"].unique()
        self._run_map = {run_id: i for i, run_id in enumerate(self.get_run_ids())}

        self.state_keys = (
            self._state["state_id"].apply(str) + "__" + self._state["state_type"]
        ).unique()

        self._node_map = {node_id: i for i, node_id in enumerate(self.state_keys)}

        self.cache_state()

    def cache_state(self) -> tuple:
        states = self._state
        n_runs = self.n_runs()
        n_steps = self.n_steps()
        state_size = self.state_size()
        state_vars = self.get_state_variables()
        n_vars = len(state_vars)

        self._state_nd = np.empty((n_runs, n_steps, state_size, n_vars))

        # node_ids = states["state_id"].unique()
        # node_map = {node_id: i for (i, node_id) in enumerate(node_ids)}

        runs_df = states.groupby(["run_id", "state_id", "state_type"])
        for (run_id, state_id, state_type), run_df in runs_df:

            i_run = self._run_map[run_id]
            i_node = self._node_map[f"{state_id}__{state_type}"]

            for i_var, var in enumerate(state_vars):
                self._state_nd[i_run, :, i_node, i_var] = run_df.loc[:, var]

    def get_input_variables(self) -> tuple:
        """Get the input keys available."""
        return self._input_keys

    def get_output_variables(self) -> tuple:
        """Get the input output available."""
        return self._output_keys

    def get_targets(self) -> tuple:
        """Get the target keys available."""
        return (*self.get_input_variables(), *self.get_output_variables())

    def get_state_variables(self) -> tuple:
        """Get the state variables available."""
        return self._state_keys

    def get_run_ids(self) -> tuple:
        """Get the ids of the experiment runs in the dataset."""
        return self._run_keys

    def n_runs(self) -> int:
        return len(self._inputs.groupby("run_id"))

    def n_steps(self) -> int:
        return self._inputs.groupby("run_id").size()[0]

    def state_size(self) -> int:
        return len(self.state_keys)

    def get_target(self, target_key, run_id) -> pd.Series:
        "Get a target signal as pandas Series by the target key."
        assert (
            target_key in self.get_targets()
        ), f"{target_key} not in available targets."
        source = self._inputs if target_key.startswith("input_") else self._outputs
        source = source.groupby("run_id").get_group(run_id)
        target_series = source[target_key]
        target_series.index = source["time"]
        return target_series

    def get_state(self, state_key, run_id) -> pd.DataFrame:
        "Get the entire reservoir state of variable as pandas DataFrame by the state key."
        assert (
            state_key in self.get_state_variables()
        ), f"{state_key} not in available state variables."

        i_run = self._run_map[run_id]
        i_state = self.get_state_variables().index(state_key)

        return self._state_nd[i_run, :, :, i_state]

    def __repr__(self) -> str:
        return (
            f"Dataset properties:\n"
            f"\tn_runs:     {self.n_runs():>3}\n"
            f"\tn_steps:    {self.n_steps():>3}\n"
            f"\tstate_size: {self.state_size():>3}\n"
            f"\nAvailable targets: \n\t{', '.join(self.get_targets())}\n"
            f"\nAvailable state variables: \n\t{', '.join(self.get_state_variables())}\n"
        )
