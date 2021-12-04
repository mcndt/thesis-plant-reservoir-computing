import sys
import arrow
import logging
from multiprocessing import Process, Pool, log_to_stderr, get_logger
from src.run.sim_process import run_experiment, check_path

def start_sim_range(path, start, end, length):

	configure_logging()

	# prepare process arguments for each experiment run
	args= []
	for i, r in enumerate(arrow.Arrow.range('day', start, end.shift(days=-(length-1)))):
		interval = r, r.shift(days=+(length-1), hours=+23)
		exp_name = f'exp_{interval[0].format("YYYY-MM-DD")}_{interval[1].format("YYYY-MM-DD")}'
		args.append((path, exp_name, interval))

	for arg in args:
		print(arg)

	worker_pool = Pool()
	result = worker_pool.starmap(run_experiment, args)  # starmap unpacks args with *
	worker_pool.close()
	worker_pool.join()

	print(f'Processed failed: {len(args) - sum(result)}')


def configure_logging():	
	log_to_stderr()
	logger = get_logger();
	logger.setLevel(logging.INFO)


if __name__ == '__main__':

	if len(sys.argv) != 5:
		print(f'Usage: python start_sim_range.py path/to/experiment/ <start_day as YYYY-MM-DD> <end_day> <duration in days> <step_size>'
					f'\nExample: python start_sim_range.py ~/hydroshoot/experiment_a 2012-08-01 2012-08-01 1')
		exit()

	path, sdate, edate, length = sys.argv[1:]
	check_path(path)
	sdate = arrow.get(sdate, 'YYYY-MM-DD')
	edate = arrow.get(edate, 'YYYY-MM-DD')
	length = int(length)

	start_sim_range(path, sdate, edate, length)