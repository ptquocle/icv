import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import config
from data_loader import get_dataloaders
from model import DenseNet14

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    _, val_loader = get_dataloaders()
    model = DenseNet14().to(device)
    checkpoint_path = "checkpoints/best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Successfully loaded {checkpoint_path}")

    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = torch.sigmoid(model(images)) 
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_outputs = torch.cat(all_outputs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print("\nPer-class AUROC:")
    results = []
    for i, class_name in enumerate(config.CLASSES):
        try:
            score = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            results.append({"Condition": class_name, "AUROC": round(score, 4)})
        except ValueError:
            results.append({"Condition": class_name, "AUROC": None}) 

    df_results = pd.DataFrame(results)
    df_results = df_results.dropna() 
    df_results.to_csv("class_performance.csv", index=False)
    
    print("Eval done")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    evaluate()
