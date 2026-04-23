#!/bin/bash
#PBS -N mimic_train
#PBS -q qgpu01
#PBS -l nodes=1:ppn=4 -W x=GRES:gpu@1
#PBS -l walltime=48:00:00
#PBS -l mem=32gb

cd /ubda/home/24116736d/projects/mimic

export LD_LIBRARY_PATH=/ubda/home/24116736d/.conda/envs/icv/lib:$LD_LIBRARY_PATH

PYTHONFAULTHANDLER=1 /ubda/home/24116736d/.conda/envs/icv/bin/python -u train.py > /ubda/home/24116736d/projects/mimic/my_training_log.txt 2>&1
