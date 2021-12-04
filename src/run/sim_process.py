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

from shutil import copytree, rmtree, ignore_patterns, copyfile
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
	
	try:
		# change the simulation time range
		if interval is not None:
			start, end = interval
			change_simulation_interval(tmp_dir, start, end)

		# run the simulation
		if exp_name is None:
			exp_name = uuid4()

		# kwargs = {}
		# if show_output:
		# 	kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE

		run_simulation(tmp_dir, sim_name=exp_name)

		# process simulation results
		try:
			process_experiment(tmp_dir, exp_name, logger=logger)
			success = True
		except Exception as e:
			logger.error(f'Error processing simulation results: \n\n{e}')
	except KeyboardInterrupt:
		logger.info(f'Keyboard interrupt. Cleaning up tmp files...')
	finally:
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


def run_simulation(path, sim_name=None):
	"""Start the simulation at the given path (path must contain sim.py script)
	and blocks until the simulation is completed. 
	
	sim_name keyword argument is used for more descriptive logging."""
	logger = get_logger()
	
	command = f"""
				cd {path};
				source ~/miniconda3/etc/profile.d/conda.sh;
				conda activate hydroshoot;
				python sim.py;
			  """

	sim_logs = os.path.join(path, 'output.txt')

	try:
		with open(sim_logs, 'w') as log_file:
			logger.info(f'[{sim_name}] Running simulation process...')
			# subprocess.run() blocks until finished
			p = subprocess.run(command, shell=True, executable='/bin/bash', stdout=log_file, stderr=log_file)  
			p.check_returncode()
			logger.info(f'[{sim_name}] Simulation process finished.')
	except subprocess.CalledProcessError as e:
			logger.info(f'[{sim_name}] Simulation process failed: \n\n{e}')
	except Exception as e:
		logger.info(f'Exception during process: \n\n{e}')
		raise e
	finally:
		# Copy simulation process logs to results directory
		log_dst = os.path.join(os.getcwd(), f'results/logs/')
		Path(log_dst).mkdir(parents=True, exist_ok=True)
		copyfile(sim_logs, os.path.join(log_dst, f'{sim_name}_output.txt'))

	
if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('\nWrong arguments. Expected exactly 1 argument')
		print(f'\nUsage: python -m src.run.sim_process <experiment_dir>\n')
		exit(-1)
	configure_logging()
	path = sys.argv[1]
	run_experiment(path, None, interval=None)