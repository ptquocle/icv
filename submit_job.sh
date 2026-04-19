#!/bin/bash
#PBS -N mimic_train
#PBS -q qgpu01
#PBS -l nodes=1:ppn=8 -W x=GRES:gpu@1
#PBS -l walltime=48:00:00
#PBS -l mem=64gb
cd /ubda/home/24116736d/projects/mimic
PYTHONFAULTHANDLER=1 /ubda/home/24116736d/.conda/envs/icv/bin/python train.py > /ubda/home/24116736d/projects/mimic/my_training_log.txt 2>&1
