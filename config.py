import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_CSV = os.path.join(DATA_DIR, "mimic-cxr-2.0.0-split.csv.gz")
CHEXPERT_CSV = os.path.join(DATA_DIR, "mimic-cxr-2.0.0-chexpert.csv.gz")
IMAGE_DIR = os.path.join(DATA_DIR, "files")

BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, IMAGE_SIZE, NUM_WORKERS = 32, 1e-4, 10, 224, 8

CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Pleural Other",
    "Support Devices",
    "No Finding",
]

NUM_CLASSES = len(CLASSES)
UNCERTAINTY_STRATEGY = "u-ones"
