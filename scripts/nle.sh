#!/bin/bash
#PBS -N mimic_batch_nle
#PBS -q qgpu01
#PBS -l nodes=1:ppn=4
#PBS -l walltime=100:00:00
#PBS -j oe
#PBS -o /ubda/home/24116736d/projects/mimic/logs/batch_nle_output.log

cd /ubda/home/24116736d/projects/mimic/

/ubda/home/24116736d/projects/mimic/.venv/bin/python -m ensurepip --upgrade
/ubda/home/24116736d/projects/mimic/.venv/bin/python -m pip install --only-binary=:all: pandas tqdm

/ubda/home/24116736d/projects/mimic/.venv/bin/python /ubda/home/24116736d/projects/mimic/src/batch_explain.py
