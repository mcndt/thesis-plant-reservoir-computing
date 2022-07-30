"""Defines a wrapper class for handling data from a single HydroShoot experiment run."""

import os
from json import load as load_json
from pickle import load as pload
import pandas as pd
import numpy as np

from src.model.hydroshoot.reservoir_state import ReservoirState
from src.util import get_dirs_in_directory

from typing import List


class HydroShootExperiment:
    """Wrapper class for handling data from a single HydroShoot experiment run."""

    def __init__(self, path: str):
        self.params = None
        self.inputs = None
        self.outputs = None
        self.states = None
        self.load_run(path)

    def load_run(self, path: str):
        state_file = os.path.join(path, "leaf_data.pickle")
        env_input_file = os.path.join(path, "env_input.csv")
        plant_output_file = os.path.join(path, "plant_outputs.csv")
        param_file = os.path.join(path, "params.json")
        self._load_params(param_file)
        self._load_outputs(plant_output_file)
        self._load_inputs(env_input_file)
        self._load_states(state_file)

    def _load_states(self, path: str):
        with open(path, "rb") as file:
            data = pload(file)
            assert isinstance(data, dict)
            self.states = ReservoirState(data)

    def _load_inputs(self, path: str):
        # NOTE: requires outputs to be loaded first!
        env_input = pd.read_csv(path)
        env_input["time"] = pd.to_datetime(
            env_input["time"], format="%Y-%m-%d %H:%M:%S"
        )
        # start = experiment.params['simulation']['sdate']
        # end = experiment.params['simulation']['edate']
        start = self.outputs["time"].iloc[0]
        end = self.outputs["time"].iloc[-1]
        env_input = env_input[(start <= env_input["time"]) & (env_input["time"] <= end)]
        self.inputs = env_input

    def _load_outputs(self, path: str):
        plant_output = pd.read_csv(path)
        plant_output["time"] = pd.to_datetime(
            plant_output["time"], format="%Y-%m-%d %H:%M:%S"
        )
        self.outputs = plant_output

    def _load_params(self, path: str):
        with open(path) as file:
            params = load_json(file)
            assert isinstance(params, dict)
            self.params = params

    def get_input_variables(self) -> tuple:
        """Get the input keys available."""
        return tuple(self.inputs.loc[:, self.inputs.columns != "time"].columns)

    def get_output_variables(self) -> tuple:
        """Get the input keys available."""
        return tuple(self.outputs.loc[:, self.outputs.columns != "time"].columns)

    def get_targets(self) -> tuple:
        """Get the target keys available."""
        target_keys = []
        target_keys += [f"input_{k}" for k in self.get_input_variables()]
        target_keys += [f"output_{k}" for k in self.get_output_variables()]
        return (*target_keys,)

    def get_state_variables(self) -> tuple:
        return self.states.get_variables()

    def get_target(self, target) -> np.ndarray:
        """Get an input our output by key. Explicitly select the scope with
        scope kwarg ('inputs' or 'outputs')"""
        if target.startswith("input_") and target[6:] in self.get_input_variables():
            return self.inputs[target[6:]].to_numpy()
        elif target.startswith("output_") and target[7:] in self.get_output_variables():
            return self.outputs[target[7:]].to_numpy()
        else:
            raise KeyError(f"No target of name '{target}'")

    def n_steps(self) -> int:
        return self.states.n_steps()

    def state_size(self) -> int:
        return self.states.state_size()

    def __repr__(self):
        return f"HydroShootExperiment(n_steps={self.n_steps()}, state_size={self.state_size()})"


def load_runs(path) -> List[HydroShootExperiment]:
    """Loads all the experiment runs present at the given path."""
    runs = []
    run_dirs = get_dirs_in_directory(path)
    for d in run_dirs:
        run_path = os.path.join(path, d)
        run = HydroShootExperiment(run_path)
        runs.append(run)
    return runs
