import numpy as np
import arrow
from typing import List, Dict, Tuple
from openalea.mtg.aml import *
from pickle import load
import os
from collections import defaultdict

FSPM_state_store = Dict[str, Dict[int, np.ndarray]]
prop_names = ['An', 'Ei', 'u', 'Tlc', 'Flux', 'Eabs', 'gs', 'E', 'gb', 'psi_head', 'Ci', 'FluxC', 'par_photo.dHd']

def get_time_series_for_class(mtg_path: str, cls: str, all_props=False) -> Tuple[List, FSPM_state_store]:
  """Loads time series data for every property name and vertex id combination
    passed from the MTG files found at mtg_path (non-recursive).
    
    Returns a tuple: (timestamps, store)
      timestamps: list of datetimes with which each time series in the store corresponds
      store: stores time series in the format: { property_name : { vertex_id : [Number] } }

    If all=False, only the predefined property names are returned. See source file for full list.
  """
  mtg_files = get_files_in_directory(mtg_path, extension='pckl')
  timestamps = [arrow.get(f[3:-4], 'YYYYMMDDHHmmss') for f in mtg_files]
  store = defaultdict(lambda: defaultdict(lambda: []))  # level 0 = dict, level 1 = dict, level 2 = list

  for file in mtg_files:
    path = os.path.join(mtg_path, file)
    mtg = load_mtg(path)
    Activate(mtg)
    vertex_ids = get_class_ids(mtg, cls)
    property_names = get_class_properties(mtg, vertex_ids, cls)
    for prop in property_names:
      data = get_property_of_vertices(mtg, prop, vertex_ids)  # returns a {vid : value} dict
      for (vid, value) in data.items():
        store[prop][vid].append(value)

  # Photosynthesis data needs to be unpacked from its dictionary shape
  if 'par_photo' in store:
    vertices = store['par_photo']
    photo_store = defaultdict(lambda: defaultdict(lambda: []))  # level 0 = dict, level 1 = dict, level 2 = list
    for (vid, series) in vertices.items():
      for p_prop in series:
        flat_dict = flatten_dict(p_prop, parent_key='par_photo', sep='.')
        for (k, v) in flat_dict.items():
          photo_store[k][vid].append(v)
    store = {**store, **photo_store}

  # Sanitize the dictionary for serialization by replacing the default dicts by dicts and filter out props if necessary
  store = { k : {k1 : v1 for (k1, v1) in v.items()} for (k, v) in store.items() if (all_props or (k in prop_names)) }
  
  # Check for data conformity: the shape of the found data matches the requested shape         
  assert(len(store) == len(prop_names))
  for v_dict in store.values():
    assert(len(v_dict) == len(vertex_ids))
    for series in v_dict.values():
      assert(len(series) == len(timestamps))
        
  return timestamps, store


def get_files_in_directory(path, extension=None):
  """get all files in the given path (non-recursive). 
  Optionally, filter by file extension."""
  files = []
  for (dirpath, dirnames, filenames) in os.walk(path):
    files.extend([f for f in filenames if not extension or f.endswith(extension)])
    break  # only explore top level directory
  return files


def load_mtg(file):
  """Loads the MTG object from file path. 
  Returns the graph object and discards the scene."""
  with open(file, 'rb') as f:
    g, _ = load(f)
    return g
    

def get_class_ids(mtg, cls: str):
  """Returns the vertex indices of all the nodes of the given class in the MTG object."""
  return [vid for vid in mtg.vertices() if Class(vid) == cls]
          
    
def get_class_properties(mtg, vertices: List[int], cls: str):
  """Returns a list of all property names that appear on vertices of given class in the MTG object."""
  properties = set()
  for vid in vertices:
    prop_names = set(mtg.get_vertex_property(vid).keys())
    properties = properties.union(prop_names)
  return list(properties)


def get_property_of_vertices(mtg, property_name, vertex_ids):
  """Get the property values for all vertices in the leaf_ids list.
  Returns a dictionary with (vid, value) pairs."""
  values = mtg.property(property_name)
  return {k : v for (k, v) in values.items() if k in vertex_ids}


def _flatten_dict_gen(d, parent_key, sep):
  # Source: https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/  
  for k, v in d.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, dict):
      yield from flatten_dict(v, new_key, sep=sep).items()
    else:
      yield new_key, v


def flatten_dict(d, parent_key='', sep='.'):
  return dict(_flatten_dict_gen(d, parent_key, sep))
