#!/bin/bash
#PBS -N mimic_eval
#PBS -q qgpu01
#PBS -l nodes=1:ppn=2 -W x=GRES:gpu@1
#PBS -l walltime=03:00:00
#PBS -l mem=16gb

cd /ubda/home/24116736d/projects/mimic

source activate icv

export LD_LIBRARY_PATH=/ubda/home/24116736d/.conda/envs/icv/lib:$LD_LIBRARY_PATH

python evaluate.py > evaluation_results.txt 2>&1
