from CNN_large import CNN3D
from small_CNN import Simple_CNN
from dataloader import FMRIDataset, DataLoader3D
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os

small_name = 'small'
large_name = 'large'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
n_class0 = 10088
n_class1 = 164
total = n_class0 + n_class1

weight_class0 = total / (2 * n_class0)
weight_class1 = total / (2 * n_class1)

weight_tensor = torch.tensor([weight_class0, weight_class1]).to(device)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)


def validate(model, val_loader, criterion, device):
    print("Validating...")
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):

            inputs, labels = inputs.to(device), labels.to(device)


            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item() * inputs.size(0)
            predicted = torch.argmax(outputs, dim=1).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.append(labels.cpu())
            all_predicted.append(predicted.cpu())


    avg_loss = val_loss / total
    f1 = f1_score(torch.cat(all_labels), torch.cat(all_predicted), average='binary', zero_division=0)
    roc_auc = roc_auc_score(torch.cat(all_labels), torch.cat(all_predicted))


    return avg_loss, f1, roc_auc, all_labels, all_predicted

def train(model, path, train_loader, val_loader, name, criterion, optimizer, device, num_epochs=50):
    print(f"Training {name}...")
    best_val_loss = 30
    patience = 8
    patience_counter = 0
    min_delta = 0.0001
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=3)
    history = {'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_f1': [], 'val_roc_auc': [], 'lrs': []}


    for epoch in range(num_epochs): 
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

        for i, batch in enumerate(train_bar):

            inputs, labels = batch
            
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item() * inputs.size(0)
            
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        

        avg_train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_f1, val_roc_auc, all_labels, all_predicted = validate(model, val_loader, criterion, device)
       

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val ROC AUC: {val_roc_auc:.4f}, LR: {current_lr:.6f}')
        

        if (val_loss + min_delta) < best_val_loss:
            best_val_loss = val_loss
            output_path = 'path/to/output'
            output_dir = output_path + path
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, name + '_best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Saving model. Current best val loss: {best_val_loss:.4f}")
            patience_counter = 0
            confusion = confusion_matrix(torch.cat(all_labels), torch.cat(all_predicted))
            report = classification_report(torch.cat(all_labels), torch.cat(all_predicted), target_names=['Not DMN', 'DMN'], zero_division=0)
        else:
            patience_counter += 1
            print(f"No improvement in validation loss. Patience counter: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_roc_auc'].append(val_roc_auc)
        history['lrs'].append(current_lr)

    return history, confusion, report

def save_history(history, name, path):

    output_path = 'path/to/output' + path
    os.makedirs(output_path, exist_ok=True)

    save_path = os.path.join(output_path, f'{name}_history.dict')

    import pickle

    with open(save_path, 'wb') as f:
        pickle.dump(history, f)



if __name__ == "__main__":
    csv_path = 'path/to/.csv'

    small_path = 'path/to/output_small'
    large_path = 'path/to/output:ì_large'
    
    # Create dataset and dataloaders
    dataset = FMRIDataset(csv_path)
    train_loader, val_loader, test_loader = DataLoader3D.create_loaders(csv_path, batch_size=16)

    # Train small CNN

    small_model = Simple_CNN(num_classes=2, dropout_rate=0.5).to(device)
    optimizer_small = torch.optim.Adam(small_model.parameters(), lr=0.001)

    history_small, small_confusion, small_report = train(small_model, small_path, train_loader, val_loader, small_name, criterion, optimizer_small, device)
    
    # Train large CNN
    large_model = CNN3D(num_classes=2, dropout_p=0.5).to(device)
    optimizer_large = torch.optim.Adam(large_model.parameters(), lr=0.001)

    history_large, large_confusion, large_report = train(large_model, large_path, train_loader, val_loader, large_name, criterion, optimizer_large, device)

    save_history(history_small, small_name, small_path)

    save_history(history_large, large_name, large_path))