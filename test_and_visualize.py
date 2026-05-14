import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import config
from data_loader import get_dataloaders
from model import DenseNet14

def test_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    _, _, test_loader = get_dataloaders()
    
    model = DenseNet14().to(device)
    checkpoint_path = "checkpoints/best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Successfully loaded {checkpoint_path}")

    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = torch.sigmoid(model(images))
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_outputs = torch.cat(all_outputs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    results = []
    
    plt.figure(figsize=(10, 8))
    print("\nPer-class AUROC on test set:")

    for i, class_name in enumerate(config.CLASSES):
        try:
            score = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            results.append({"Condition": class_name, "AUROC": round(score, 4)})
            
            fpr, tpr, _ = roc_curve(all_labels[:, i], all_outputs[:, i])
            plt.plot(fpr, tpr, label=f"{class_name} (AUC = {score:.4f})")
        except ValueError:
            # cases where a class might only have 0s or 1s in the batch/test set
            results.append({"Condition": class_name, "AUROC": None})

    df_results = pd.DataFrame(results)
    df_results = df_results.dropna()
    df_results.to_csv("test_performance.csv", index=False)
    
    print("\nTest evaluation done.")
    print(df_results.to_string(index=False))

    plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curves on test set')
    plt.legend(loc="lower right", fontsize='small', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.savefig("test_roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(df_results["Condition"], df_results["AUROC"], color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("AUROC")
    plt.title("Test set AUROC per class")
    plt.ylim(0, 1.05)
    for i, v in enumerate(df_results["AUROC"]):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8, rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("test_auroc_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    test_and_visualize()

