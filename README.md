# MIMIC-CXR training pipeline

This repository contains a training pipeline for the **MIMIC-CXR** dataset using a DenseNet121 architecture. It is to run on the UBDA cluster only allowing me to run on Tesla P100!!!

## Structure

| File | Description |
| :--- | :--- |
| `train.py` | Main training loop. |
| `model.py` | DenseNet121 architecture with a custom 14-class multi-label head. |
| `data_loader.py` | DataLoader using pre-split CSV files. |
| `config.py` | Hyperparameters and path configs. |
| `split_files.py` | Pre-processing script to split data without RAM overhead. |
| `submit_job.sh` | PBS script for requesting 8 CPUs, 64GB RAM, and 1 GPU. |
| `utils.py` | Helper functions. |

## Dataset
To avoid memory crashes on login nodes, the dataset is split into chunks:
1. `create_master.py` merges raw MIMIC metadata into a single master file.
2. `split_files.py` generates `mimic_train.csv` and `mimic_val.csv`.

## Config
- **Batch Size:** 32
- **Num Workers:** 8
- **Input Size:** 224x224
- **Epochs:** 10

## Usage
1. **Prepare data:** `python create_master.py | python split_files.py`
2. **Submit job:** `qsub submit_job.sh`
3. **Monitor:** `tail -f my_training_log.txt`
