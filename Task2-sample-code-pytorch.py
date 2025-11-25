"""
CNN classifier for image classification using PyTorch
"""
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class PokemonClassificationModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


def train_model(model, train_loader, optimizer, loss_fn, device, num_classes):
    train_loss = 0.0
    model.train()
    all_predictions = []
    all_labels = []

    for (img, label) in tqdm(train_loader, ncols=80, desc='Training'):
        img, label = img.to(device, dtype=torch.float), label.to(device, dtype=torch.long)
        optimizer.zero_grad()
        logits = model(img)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        prediction = logits.argmax(axis=1)
        all_predictions.extend(prediction.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_predictions)
    return train_loss / len(train_loader), train_acc


def validate_model(model, val_loader, loss_fn, device, num_classes):
    model.eval()
    valid_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for (img, label) in tqdm(val_loader, ncols=80, desc='Valid'):
            img, label = img.to(device, dtype=torch.float), label.to(device, dtype=torch.long)
            logits = model(img)
            loss = loss_fn(logits, label)
            valid_loss += loss.item()
            
            prediction = logits.argmax(axis=1)
            all_predictions.extend(prediction.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_predictions)
    return valid_loss / len(val_loader), val_acc


def test_model(model, test_loader, device, class_names):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for (img, label) in tqdm(test_loader, ncols=80, desc='Testing'):
            img, label = img.to(device, dtype=torch.float), label.to(device, dtype=torch.long)
            logits = model(img)
            prediction = logits.argmax(axis=1)
            all_predictions.extend(prediction.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_predictions)
    test_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    test_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    test_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return test_accuracy, test_precision, test_recall, test_f1, all_predictions, all_labels


def evaluate_different_resolutions(model, base_path, device, class_names):
    resolutions = [32, 64, 128, 256, 512]
    results = {}
    
    for res in resolutions:
        print(f"\nEvaluating at resolution: {res}×{res}")
        
        transform = transforms.Compose([
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        dataset = ImageFolder(root=base_path, transform=transform)
        torch.manual_seed(42)
        _, test_set, _ = random_split(dataset, [0.7, 0.2, 0.1])
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        
        accuracy, precision, recall, f1, _, _ = test_model(model, test_loader, device, class_names)
        
        results[res] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return results


if __name__ == "__main__":
    print("CNN Classifier for Pokemon Classification (PyTorch)")
    print("=" * 60)

    print("Setting up data transforms and loading dataset...")
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    data = ImageFolder(root='PokemonData/PokemonData', transform=train_transform)
    torch.manual_seed(42)
    train_set, test_set, val_set = random_split(data, [0.7, 0.2, 0.1])
    
    print(f"Dataset loaded: {len(data)} total samples")
    print(f"Classes: {data.classes}")
    print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model = PokemonClassificationModel(num_classes=len(data.classes)).to(device)
    print(f"Model created and moved to {device}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nCNN Architecture:")
    print("Block 1: Conv2d(3→32) + BatchNorm + ReLU + MaxPool2d")
    print("Block 2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool2d")
    print("Block 3: Conv2d(64→128) + BatchNorm + ReLU + MaxPool2d")
    print("Block 4: Conv2d(128→256) + BatchNorm + ReLU + MaxPool2d")
    print("Adaptive Pooling: AdaptiveAvgPool2d(4×4)")
    print("FC Layers: 4096 → 512 → 256 → 10 (with Dropout 0.5)")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_acc = 0.0
    epochs = 50
    
    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, loss_fn, device, len(data.classes))
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_loss, val_acc = validate_model(model, val_loader, loss_fn, device, len(data.classes))
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_pytorch_cnn_model.pth')
            print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} * (Best)")
        else:
            print(f"Epoch {epoch+1:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    model.load_state_dict(torch.load('best_pytorch_cnn_model.pth'))
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    print("\nEvaluating on test set...")
    test_acc, test_prec, test_rec, test_f1, predictions, true_labels = test_model(model, test_loader, device, data.classes)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    plt.figure(figsize=(18, 5))
    plt.suptitle('CNN Analysis - PyTorch Implementation', fontsize=16, y=1.02)
    
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Figure 1a: Training Curves - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Figure 1b: Training Curves - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 3)
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=data.classes, yticklabels=data.classes)
    plt.title('Figure 1c: Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(cm_norm, 0)
    max_confusion_idx = np.unravel_index(np.argmax(cm_norm), cm_norm.shape)
    most_confused_pair = (data.classes[max_confusion_idx[0]], data.classes[max_confusion_idx[1]])
    confusion_rate = cm_norm[max_confusion_idx]
    
    print(f"\nMost frequently confused pair: {most_confused_pair[0]} -> {most_confused_pair[1]}")
    print(f"Confusion rate: {confusion_rate:.4f}")
    
    print("\n" + "=" * 60)
    print("Evaluating at different resolutions")
    print("=" * 60)
    
    resolution_results = evaluate_different_resolutions(model, 'PokemonData/PokemonData', device, data.classes)
    
    plt.subplot(1, 4, 4)
    resolutions = list(resolution_results.keys())
    accuracies = [resolution_results[res]['accuracy'] for res in resolutions]
    plt.plot(resolutions, accuracies, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Image Resolution (pixels)')
    plt.ylabel('Test Accuracy')
    plt.title('Figure 1d: Resolution Impact Analysis')
    plt.grid(True, alpha=0.3)
    for i, acc in enumerate(accuracies):
        plt.annotate(f'{acc:.3f}', (resolutions[i], acc), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('Figure_1_CNN_Training_History.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Figure 1: CNN Training History saved as Figure_1_CNN_Training_History.png")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(true_labels, predictions, target_names=data.classes))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Most confused pair: {most_confused_pair}")
    print(f"Best resolution: {max(resolution_results.keys(), key=lambda x: resolution_results[x]['accuracy'])} pixels")
    print(f"Model saved as: best_pytorch_cnn_model.pth")
