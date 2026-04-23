# MIMMIC-CXR
This repository contains a pipeline for multi-label classification of thoracic pathologies using the **MIMIC-CXR** dataset, for URIS progress report.

---

## Project structure
```text
├── checkpoints/          # Saved model weights (.pth)
├── data/                 # Dataset directory 
├── config.py             # Hyperparameters
├── data_loader.py        # Data loader
├── model.py              # DenseNet
├── train.py              # Main training script 
├── evaluate.py           # Per-class performance eval
├── visualize.py          # Script to generate loss/AUROC plots
├── submit_job.sh         # PBS script for training
├── evaluate.sh           # PBS script for evaluation
├── utils.py              # Helpers
├── create_master.py      # Merges raw metadata into one
└── split_files.py        # Generates `mimic_train.csv` and `mimic_val.csv`
```

## Getting started

### 1. Environment Setup
```bash
conda create -n icv python=3.7
conda activate icv
pip install torch torchvision tqdm pandas scikit-learn matplotlib
```

### 2. Training on the cluster
Prepare data:
```bash
python create_master.py
python split_files.py
```

To submit a training job:

```bash
qsub submit_job.sh
```

### 3. Evaluation
Once training is complete, run:

```bash
qsub evaluate.sh
```

This generates `class_performance.csv`, containing the AUROC score for each of the 14 pathologies.
