from dataloader import FMRIDataset, DataLoader3D
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import average_precision_score
from CNN_training import plot_cm, save_cm_and_cr, validate
from small_CNN import Simple_CNN
from CNN_large import CNN3D

def test(model, test_loader, criterion, device):
    print("Testing...")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]   # probabilità della classe DMN
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_probs.extend(probs.detach().cpu())
            all_labels.extend(labels.cpu())
            all_predicted.extend(predicted.cpu())

    avg_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct / total if total > 0 else 0
    f1 = f1_score(all_labels, all_predicted, average='binary', zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    return avg_test_loss, test_accuracy, all_labels, all_predicted, f1, roc_auc, all_probs

if __name__ == "__main__":

    csv_path = 'path/to/.csv'

    small_name = 'small_test'
    large_name = 'large_test'

    dataset = FMRIDataset(csv_path)
    train_loader, val_loader, test_loader = DataLoader3D.create_loaders(csv_path, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_class0 = 10088
    n_class1 = 164
    total = n_class0 + n_class1

    weight_class0 = total / (2 * n_class0)
    weight_class1 = total / (2 * n_class1)

    weight_tensor = torch.tensor([weight_class0, weight_class1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    large_model = CNN3D(num_classes=2, dropout_p = 0.5)
    small_model = Simple_CNN(num_classes=2, dropout_rate = 0.5)

    large_model.load_state_dict(torch.load('path/to/large_best_model.pth', map_location=device))
    large_model.to(device)

    avg_test_loss, test_accuracy, all_labels, all_predicted, f1, roc_auc, all_probs = test(large_model, test_loader, criterion, device)
    
    auprc = average_precision_score(all_labels, all_probs)
    print(f'auprc:{auprc}, roc_auc:{roc_auc}')

    small_model.load_state_dict(torch.load('path/to/small_best_model.pth', map_location=device))
    small_model.to(device)

    avg_test_loss, test_accuracy, all_labels, all_predicted, f1, roc_auc, all_probs = test(small_model, test_loader, criterion, device)

    auprc = average_precision_score(all_labels, all_probs)
    print(f'auprc: {auprc}, roc_auc: {roc_auc}')
