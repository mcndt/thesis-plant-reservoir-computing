import sys
import os
import pandas as pd
import numpy as np
import arrow

from pickle import dump
from typing import Dict
from pathlib import Path
from shutil import copyfile

from extract_vertex_properties import get_time_series_for_class

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


def process_experiment(path: str, destination: str=None) -> str:
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
  print('Extracted time series: ')
  print(f'\tfrom {timestamps[0].format("YYYY-MM-DD HH:mm")} to {timestamps[-1].format("YYYY-MM-DD HH:mm")}')
  print(f'\t{len(timestamps):<4} steps')
  print(f'\t{len(store):<4} properties')
  print(f'\t{len(list(store.values())[0]):<4} state variables')

  # Create destination folder if necessary
  if destination is None:
    experiment_name = arrow.now().format('YYYY-MM-DD_HHmm_SSSSSSSSSS')  # Add a lot of detail to the second mark to ensure it will be a unique directory :)
  else:
    experiment_name = destination
  destination = os.path.join(os.getcwd(), f'results/{experiment_name}')
  Path(destination).mkdir(parents=True, exist_ok=True)

  # Write all results to disk
  inputs.to_csv(os.path.join(destination, 'env_input.csv'), index=False)
  output_series.to_csv(os.path.join(destination, 'plant_outputs.csv'), index=False)
  copyfile(paths['params'], os.path.join(destination, 'params.json'))
  with open(os.path.join(destination, 'leaf_data.pickle'), 'wb') as f:
    dump(store, f)

  return destination

if __name__ == "__main__":
  path = sys.argv[1]
  destination = sys.argv[2] if len(sys.argv) > 2 else None
  if not os.path.isdir(path):
    print(f"'{path}' is not a valid path.")
    exit()
  output_dir = process_experiment(path, destination=destination)
  print(f'Results written to {output_dir}\n')