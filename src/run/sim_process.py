"""
This script serves as a self-contained process for running a simulation 
and processing the intermediate states in order to perform self-cleanup.
"""
import os
import sys
import json
import arrow
import logging
import subprocess

from shutil import copytree, rmtree, ignore_patterns
from pathlib import Path
from uuid import uuid4
from multiprocessing import current_process, get_logger, log_to_stderr


from typing import Tuple
from arrow.arrow import Arrow

from src.post_processing.process_experiment import process_experiment
from src.constants import HYDROSHOOT_DATETIME_FORMAT

def run_experiment(path, exp_name, interval: Tuple[Arrow, Arrow]=None):
	"""Returns False if the process failed, True otherwise.
	
	To change the experiment's time range, give a datetime-like
	tuple of (start, end) times in the range keyword arg.
	"""
	success=False
	# configure_logging()
	logger = get_logger()

	# create a temp copy of the experiment
	check_path(path)
	tmp_dir = make_tmp_dir(path)
	
	# change the simulation time range
	if interval is not None:
		start, end = interval
		change_simulation_interval(tmp_dir, start, end)

	# run the simulation
	try:
		run_simulation(tmp_dir, sim_name=exp_name, verbose=False)
		# process simulation results
		if exp_name is None:
			exp_name = uuid4()
		try:
			process_experiment(tmp_dir, exp_name, logger=logger)
			success = True
		except Exception as e:
			logger.error(f'Error processing simulation results: \n\n{e}')

	except Exception as e:
		logger.error(f"Error running simulation: \n\n{e}")

	# cleanup
	remove_tmp_dir(tmp_dir)
	return success


def configure_logging():
	"""Configures the logging for this (sub)process to log up to INFO level."""
	log_to_stderr()
	logger = get_logger()
	logger.setLevel(logging.INFO)


def check_path(path):
	"""Raises exceptions if the path is invalid or files are missing."""
	if not os.path.isdir(path):
		raise NotADirectoryError(f'Path is invalid: "{path}"')

	sim_script = os.path.join(path, 'sim.py')
	if not os.path.isfile(sim_script):
		raise FileNotFoundError(f'Cannot find simulation script in path: {sim_script}')
		

def make_tmp_dir(path):
	"""Copies the contents at the path to a temporary directory and 
	returns the newly created path."""
	tmp_path_name = f'tmp_{uuid4()}'
	tmp_path = Path(path).parent.absolute().joinpath(tmp_path_name)
	copytree(path, tmp_path, ignore=ignore_patterns('output', '*.ipynb', '.ipynb_checkpoints'))
	get_logger().info(f'Created temp directory: {tmp_path}')
	return tmp_path


def remove_tmp_dir(path):
	"""Deletes the temporary folder at path."""
	rmtree(path)
	get_logger().info(f'Removed temp directory: {path}')


def change_simulation_interval(path, start, end):
	"""Modifies the params.json in the experiment path to the 
	new start and end time."""
	params_path = os.path.join(path, 'params.json')
	if not os.path.isfile(params_path):
		raise FileNotFoundError(f"Simluation directory has no params file: '{params_path}'")
	
	with open(params_path) as f:
		params = json.load(f)
		params['simulation']['sdate'] = start.format(HYDROSHOOT_DATETIME_FORMAT)
		params['simulation']['edate'] = end.format(HYDROSHOOT_DATETIME_FORMAT)

	with open(params_path, 'w') as f:
		json.dump(params, f, indent=2)

	get_logger().info(f'Wrote simulation interval {start.format(HYDROSHOOT_DATETIME_FORMAT)}-{end.format(HYDROSHOOT_DATETIME_FORMAT)} to params.json')


def run_simulation(path, sim_name=None, verbose=False):
	"""Start the simulation at the given path (path must contain sim.py script)
	and blocks until the simulation is completed. 
	
	sim_name keyword argument is used for more descriptive logging."""
	
	command = f"""
				cd {path};
				source ~/miniconda3/etc/profile.d/conda.sh;
				conda activate hydroshoot;
				python sim.py;
			  """

	logger = get_logger()
	logger.info(f'Starting simulation "{sim_name}"...')

	run_kwargs = {}
	if verbose:
		run_kwargs['stdout'] = sys.stdout
		run_kwargs['stderr'] = sys.stderr
	else:
		run_kwargs['capture_output'] = True

	# subprocess.run() blocks until finished
	p = subprocess.run(command, shell=True, executable='/bin/bash', **run_kwargs)  
	logger.info(f'Simulation finished: "{sim_name}".')
	if not (__name__ == '__main__' or verbose):
		stderr = p.stderr.decode().strip();
		if len(stderr) > 0:
			logger.warn(stderr)
		

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('Wrong arguments. Expects exactly 1 argument')
		print(f'Usage: python {__file__} path/to/experiment/')
		exit(-1)
	path = sys.argv[1]
	run_experiment(path, None)