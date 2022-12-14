# Queue and run multiple experiments using a multiprocessing
# worker pool. Experiments are defined in a JSON file.
#
# JSON file should contain a JSON list of experiment objects.
#
# For example:
#
# [
#   {
#     "path": "path/to/experiment/",
#     "sdate": "2012-08-01",
#     "edate": "2012-08-31",
#     "length": 5,
#   }
# ]
import os
import sys
import json
import arrow
import argparse
from multiprocessing import Pool, log_to_stderr, get_logger
from src.run.sim_process import run_experiment, check_path, configure_logging


def get_args(fpath: str):
    """Loads the JSON object at fpath and generates an iterable
	of args for src.run.sim_process"""
    args = []
    with open(fpath) as f:
        experiments = json.load(f)
        if not isinstance(experiments, list):
            raise TypeError(
                "Experiment json file should contain a list of experiment objects."
            )

        for exp in experiments:
            start, end, length = exp["sdate"], exp["edate"], exp["length"]
            start = arrow.get(start, "YYYY-MM-DD")
            end = arrow.get(end, "YYYY-MM-DD")
            length = int(length)

            extend_edate = "extend" in exp and exp["extend"]
            offset_days = 0 if extend_edate else -(length - 1)

            for i, r in enumerate(
                arrow.Arrow.range("day", start, end.shift(days=offset_days))
            ):
                interval = r, r.shift(days=+(length - 1), hours=+23)
                exp_name = f'{exp["name"]}_{i}'
                run_args = [exp["path"], exp_name, interval]

                if "meteo" in exp:
                    run_args.append(exp["meteo"])

                args.append(run_args)

    return args


def run_experiments(fpath: str):
    """Converts the JSON file at the fpath argument into a list of simulation
	arguments, and executes the simulations using a multiprocessing worker pool."""
    args = get_args(fpath)

    print("\n\n+---------------------------------+")
    print(f'Queueing {len(args)} experiment{"s" if len(args) > 2 else ""}:')
    for i, a in enumerate(args):
        print(
            f'{i:>4}.   {a[1]}    {a[2][0].format("YYYY-MM-DD HH:mm")} - {a[2][1].format("YYYY-MM-DD HH:mm")}'
        )
    print("+---------------------------------+\n\n")

    worker_pool = Pool()
    result = worker_pool.starmap(run_experiment, args)  # starmap unpacks args with *
    worker_pool.close()
    worker_pool.join()

    print(f"\n{sum(result)}/{len(args)} simulations completed successfully.")

    if sum(result) > 0:
        print("\nFailed processes:")
        for i, r in enumerate(result):
            if not r:
                print(f"\t{args[i][1]}")
        print("\nCheck logs for more details.")

    destination = os.path.join(os.getcwd(), f"results/")
    print(f"Output datasets are at {destination}")


if __name__ == "__main__":
    # create CLI argument parser
    parser = argparse.ArgumentParser(
        description=f"Run a batch of experiments using a multiprocessing worker pool."
        f"\n\nSee sample_experiments/ for examples."
    )
    parser.add_argument(
        "json_file", type=str, help="The JSON file containing the experiments to run."
    )

    # parse the arguments
    args = parser.parse_args()

    configure_logging()
    path = sys.argv[1]
    run_experiments(args.json_file)

