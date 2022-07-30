"""Defines a wrapper class around the reservoir state of a plant simulation."""

from typing import Dict
import numpy as np

FspmStateSTore = Dict[str, Dict[int, np.ndarray]]


class ReservoirState:
    """Wrapper class to store reservoir state over multiple node variables."""

    def __init__(self, states: FspmStateSTore):
        self._states = states

    def __getitem__(self, key) -> np.ndarray:
        """Get a ndarray of the state history for a specific state variable.
        State variables are ordered in ascending mtg vertex id.
          axis 0: nodes
          axis 1: steps"""

        data = np.empty((self.n_steps(), self.state_size()))
        variable_nodes = self._states[key]
        for i, series in enumerate(variable_nodes.values()):
            data[:, i] = series

        # reorder by node id
        node_ids = list(variable_nodes.keys())
        sort_key = np.argsort(node_ids)
        data = data[:, sort_key]
        return data

    def get_variables(self):
        """Return a list of available veriable keys"""
        return tuple(self._states.keys())

    def state_size(self):
        """Returns the amount of reservoir state nodes available."""
        variable = next(iter(self._states.values()))
        return len(variable)

    def n_steps(self):
        """Returns the amount of reservoir steps available."""
        variable = next(iter(self._states.values()))
        node = next(iter(variable.values()))
        return len(node)
