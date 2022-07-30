import os

from typing import List

def get_dirs_in_directory(path: str) -> List[str]:
    """get all subdirectories in the given path (non-recursive)."""
    dirs = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        dirs.extend([f for f in dirnames])
        break  # only explore top level directory
    return dirs
