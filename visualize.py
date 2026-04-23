import matplotlib.pyplot as plt
import pandas as pd
import re
import os

def plot_learning_curves(log_path):
    if not os.path.exists(log_path):
        return
    
    epochs, train_losses, val_losses, aurocs = [], [], [], []
    
    with open(log_path, 'r') as f:
        log_content = f.read()
    epoch_blocks = re.findall(r"Epoch (\d+)/\d+.*?train loss: ([\d.]+) \| val loss: ([\d.]+) \| val macro AUROC: ([\d.]+)", 
                              log_content, re.DOTALL)
    
    for epoch, t_loss, v_loss, auroc in epoch_blocks:
        epochs.append(int(epoch))
        train_losses.append(float(t_loss))
        val_losses.append(float(v_loss))
        aurocs.append(float(auroc))
    
    plt.figure(figsize=(14, 5))

    # Plot 1: training & validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train loss', marker='o', color='#1f77b4')
    plt.plot(epochs, val_losses, label='Val loss', marker='o', color='#ff7f0e')
    plt.title('Training & validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Validation AUROC
    plt.subplot(1, 2, 2)
    plt.plot(epochs, aurocs, label='Val macro AUROC', marker='s', color='#2ca02c', linewidth=2)
    plt.axhline(y=0.80, color='r', linestyle='--', label='Baseline target (0.80)')
    plt.title('Validation performance (AUROC)')
    plt.xlabel('Epoch')
    plt.ylabel('AUROC')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)
    print("Saved learning_curves.png")

def plot_class_performance(csv_path):
    if not os.path.exists(csv_path):
        print(f"Cannot find {csv_path}. Run evaluate.py first!")
        return

    df = pd.read_csv(csv_path)
    df = df.sort_values(by='AUROC', ascending=True)

    plt.figure(figsize=(10, 8))
    bars = plt.barh(df['Condition'], df['AUROC'], color='#17becf')
    
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random chance (0.5)')

    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.4f}', 
                 va='center', ha='left', fontsize=10)

    plt.title('Per-class validation AUROC (best model)')
    plt.xlabel('AUROC score')
    plt.ylabel('Condition')
    plt.xlim(0.4, 1.0) # Start at 0.4 to emphasize differences
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('class_performance_bar.png', dpi=300)
    print("Saved class_performance_bar.png")

if __name__ == "__main__":
    plot_learning_curves('my_training_log.txt')
    plot_class_performance('class_performance.csv')
