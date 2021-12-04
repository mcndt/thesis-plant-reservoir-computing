#!/bin/bash

cd $1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hydroshoot
python sim.py