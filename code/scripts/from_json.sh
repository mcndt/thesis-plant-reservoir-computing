#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hydroshoot
python -m src.run.from_json "$@"