import torch
from tqdm import tqdm
import config
from data_loader import MIMICCXRDataset
from model import DenseNet14
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        normalize])
    
    test_dataset = MIMICCXRDataset(split_name='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    model = DenseNet14().to(device)
    model.load_state_dict(torch.load("checkpoints/model.pth", map_location=device))
    model.eval()
    all_outputs, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="evaluating test set"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            all_outputs.append(outputs)
            all_labels.append(labels)
            
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    
    probs = torch.sigmoid(all_outputs).cpu().numpy()
    targets = all_labels.cpu().numpy()
    
    print("\nTEST SET RESULTS")
    aurocs = []
    for i, class_name in enumerate(config.CLASSES):
        try:
            score = roc_auc_score(targets[:, i], probs[:, i])
            aurocs.append(score)
            print(f"{class_name:<30}: {score:.4f}")
        except ValueError:
            print(f"{class_name:<30}: N/A (not enough variation in labels)")
    print(f"{'macro average':<30}: {sum(aurocs)/len(aurocs):.4f}")

if __name__ == "__main__": evaluate_model()
