#!/bin/bash -l
#SBATCH --mem=40G
#SBATCH --ntasks=8
#SBATCH --qos=normal
#SBATCH --time=360
#SBATCH --export=NONE

module load scitools
python py_process_data.py
