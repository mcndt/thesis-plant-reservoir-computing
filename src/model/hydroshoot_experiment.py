"""Defines a wrapper class for handling data from a single HydroShoot experiment run."""

import os
from json import load as load_json
from pickle import load as pload
import pandas as pd
import numpy as np

from src.model.reservoir_state import ReservoirState

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
        state_file = os.path.join(path, 'leaf_data.pickle')
        env_input_file = os.path.join(path, 'env_input.csv')
        plant_output_file = os.path.join(path, 'plant_outputs.csv')
        param_file = os.path.join(path, 'params.json')
        self._load_states(state_file)
        self._load_inputs(env_input_file)
        self._load_outputs(plant_output_file)
        self._load_params(param_file)

    def _load_states(self, path: str):
        with open(path, 'rb') as file:
            data = pload(file)
            assert(isinstance(data, dict))
            self.states = ReservoirState(data)

    def _load_inputs(self, path: str):
        env_input = pd.read_csv(path)
        env_input['time'] = pd.to_datetime(
            env_input['time'], format='%Y-%m-%d %H:%M:%S')
        self.inputs = env_input

    def _load_outputs(self, path: str):
        plant_output = pd.read_csv(path)
        plant_output['time'] = pd.to_datetime(
            plant_output['time'], format='%Y-%m-%d %H:%M:%S')
        self.outputs = plant_output

    def _load_params(self, path: str):
        with open(path) as file:
            params = load_json(file)
            assert(isinstance(params, dict))
            self.params = params

    def get_input_variables(self) -> tuple:
        """Get the input keys available."""
        return tuple(self.inputs.columns)

    def get_output_variables(self) -> tuple:
        """Get the input keys available."""
        return tuple(self.outputs.columns)

    def get_state_variables(self) -> tuple:
        return self.states.get_variables()

    def get_target(self, target, scope=None) -> np.ndarray:
        """Get an input our output by key. Explicitly select the scope with
        scope kwarg ('inputs' or 'outputs')"""
        if target in self.get_input_variables() and target in self.get_output_variables() and scope is None:
            raise KeyError(f'Target \'{target}\' is ambiguous because it appears in multiple scopes (try kwarg scope=\'outputs\' or \'inputs\')')
        if target in self.get_input_variables() and (scope is None or scope == 'inputs'):
            return self.inputs[target].to_numpy()
        elif target in self.get_output_variables() and (scope is None or scope == 'outputs'):
            return self.outputs[target].to_numpy()
        else:
            raise KeyError(f"No target of name '{target}'")

    def n_steps(self) -> int:
        return self.states.n_steps()

    def state_size(self) -> int:
        return self.states.state_size()

    def __repr__(self):
        return f'HydroShootExperiment(n_steps={self.n_steps()}, state_size={self.state_size()})'


def load_runs(path) -> List[HydroShootExperiment]:
    """Loads all the experiment runs present at the given path."""
    runs = []
    run_dirs = get_dirs_in_directory(path)
    for d in run_dirs:
        run_path = os.path.join(path, d)
        run = HydroShootExperiment(run_path)
        runs.append(run)
    return runs


def get_dirs_in_directory(path) -> List[str]:
    """get all subdirectories in the given path (non-recursive)."""
    dirs = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        dirs.extend([f for f in dirnames])
        break  # only explore top level directory
    return dirs