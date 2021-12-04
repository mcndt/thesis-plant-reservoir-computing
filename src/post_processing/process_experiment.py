import sys
import os
import pandas as pd
import numpy as np
import arrow
import logging

from pickle import dump
from typing import Dict
from pathlib import Path
from shutil import copyfile

from src.post_processing.extract_vertex_properties import get_time_series_for_class

def get_file_paths(path: str) -> Dict[str, str]:
  """Checks if the passed path contains all the required experiment inputs and outputs.
  Throws an exception if an input or output file is missing."""
  paths = {
    "input": os.path.join(path, 'meteo.input'),
    "params": os.path.join(path, "params.json"),
    "output_dir": os.path.join(path, "output/"),
    "output_series": os.path.join(path, "output/time_series.output")
  }

  for p in paths.values():
    if not os.path.exists(p):
      raise FileNotFoundError(f"No such file/directory: '{p}'")

  return paths;


def process_experiment(path: str, name: str, logger=None) -> str:
  """Takes as input the root folder of a HydroShoot experiment and 
  outputs a directory 'results/' containing the following serialized objects: 

  "results/{experiment_name}/env_input.csv (Pandas dataframe containing the environmental inputs of the experiment)
  "results/{experiment_name}/params.json" (Dictionary containing the HydroShoot params JSON used for the experiment)
  "results/{experiment_name}/plant_outputs.csv (Pandas dataframe containing the plant-scale performance metrics as computed by HydroShoot)
  "results/{experiment_name}/leaf_data.pickle" (Dictionary containing all leaf properties as time series)

  Returns the output path as string.
  """
  paths = get_file_paths(path)

  # Read environmental inputs
  inputs = pd.read_csv(paths['input'], sep=';')
  inputs = inputs[['time', 'Tac', 'u', 'hs', 'Rg']]

  # Read plant performance outputs
  output_series = pd.read_csv(paths['output_series'], sep=';')

  # Extract all leaf-level series
  timestamps, store = get_time_series_for_class(paths['output_dir'], 'L')

  if logger:
    logger.info(f'[{name}] Extracted {len(timestamps)} steps, {len(store)} properties, {len(list(store.values())[0])} state size')
  else:
    print(f'\nExtracted time series: '
          f'\n\tfrom {timestamps[0].format("YYYY-MM-DD HH:mm")} to {timestamps[-1].format("YYYY-MM-DD HH:mm")}'
          f'\n\t{len(timestamps):<4} steps'
          f'\n\t{len(store):<4} properties'
          f'\n\t{len(list(store.values())[0]):<4} state variables')

  # Create destination folder if necessary
  destination = os.path.join(os.getcwd(), f'results/{name}')
  Path(destination).mkdir(parents=True, exist_ok=True)

  # Write all results to disk
  inputs.to_csv(os.path.join(destination, 'env_input.csv'), index=False)
  output_series.to_csv(os.path.join(destination, 'plant_outputs.csv'), index=False)
  copyfile(paths['params'], os.path.join(destination, 'params.json'))
  with open(os.path.join(destination, 'leaf_data.pickle'), 'wb') as f:
    dump(store, f)

  return destination

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("1st arg: path/to/experiment/,   2nd arg: experiment name")
    exit()
  path = sys.argv[1]
  name = sys.argv[2]
  if not os.path.isdir(path):
    print(f"'{path}' is not a valid path.")
    exit()

  output_dir = process_experiment(path, name)
  print(f'Results written to {output_dir}\n')