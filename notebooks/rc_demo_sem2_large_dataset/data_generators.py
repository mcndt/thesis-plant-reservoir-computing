import sys, os
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], "../../"))
from src.model.rc_dataset import ExperimentDataset


def direct_target_generator(dataset: ExperimentDataset, target: str, run_ids: [int]):
    """Returns a generator that generates the target from the run id."""
    assert target in dataset.get_targets(), f"{target} not available in dataset."

    # preload data in numpy array for performance reasons
    data = np.empty((len(run_ids), dataset.n_steps()))
    for run_id in run_ids:
        data[run_id, :] = dataset.get_target(target, run_id).to_numpy()

    for run_id in run_ids:
        yield data[run_id, :]


def direct_reservoir_generator(
    dataset: ExperimentDataset,
    state_var: str,
    run_ids: [int],
    state_size=-1,
    random_state=None,
):
    """Returns a function that generates the reservoir from the run id."""
    assert (
        state_var in dataset.get_state_variables()
    ), f"{state_var} not available in dataset."

    if state_size > 0:
        if random_state is not None:
            np.random.seed(random_state)
        state_choice = np.random.choice(
            dataset.state_size(), size=state_size, replace=False
        )
    else:
        state_choice = slice(0, dataset.state_size())

    for run_id in run_ids:
        yield dataset.get_state(state_var, run_id)[:, state_choice]

