from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    f1_score,
    accuracy_score,
    classification_report,
    roc_auc_score
)
import numpy as np
from CNN_large import CNN3D
from small_CNN import small_CNN
from dataloader import FMRIDataset, DataLoader3D
import torch
from tqdm import tqdm
import pandas as pd
from CNN_training import plot_cm, save_cm_and_cr
import os
import joblib
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_features(model, dataloader, layer_name, device):
    activation = {}

    def hook(module, input, output):
        activation["feat"] = output.detach()

    # Find the requested layer
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook)

    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for x, _, y in tqdm(dataloader, desc="Extracting features", leave=False):
            x = x.to(device)
            _ = model(x)

            feat = activation["feat"]

            # If it's a Linear layer before ReLU
            if isinstance(layer, torch.nn.Linear):
                feat = torch.relu(feat)

            features.append(feat.cpu())
            labels.append(y)

    handle.remove()

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    return features, labels


if __name__ == "__main__":

    small_path = 'path/to/small_CNN.py'
    large_path = 'path/to/large_CNN.py'
    csv_path = 'path/to/.csv'

    results_small = []
    results_large = []    
    # Create dataset and dataloaders
    dataset = FMRIDataset(csv_path)
    train_loader, val_loader, test_loader = DataLoader3D.create_loaders(csv_path, batch_size=16)

    small_model = small_CNN(num_classes=2, dropout_rate = 0.5)
    large_model = CNN3D(num_classes=2, dropout_p = 0.5)

    small_model.load_state_dict(torch.load('path/to/small_best_model.pth', map_location=device))
    large_model.load_state_dict(torch.load('path/to/large_best_model.pth', map_location=device))

    small_model.to(device)
    small_model.eval()

    large_model.to(device)
    large_model.eval()
    X1, y1 = extract_features(small_model, train_loader, 'fc1', device)
    X2, y2 = extract_features(large_model, train_loader, 'fc.0', device)

    X1 = X1[y1 == 0]
    X2 = X2[y2 == 0]

    Z1_val, w1_val = extract_features(small_model, val_loader, 'fc1', device)
    Z2_val, w2_val = extract_features(large_model, val_loader, 'fc.0', device)

    kernel = ['rbf', 'poly', 'sigmoid']
    gamma = ['scale', 'auto']
    nu = [0.1, 0.15, 0.25, 0.5, 0.75]

    best_small = -1
    best_small_model = None
    best_large = -1
    best_large_score = None
    best_params = None
    best_params_large = None

    for k in kernel:
        for g in gamma: 
            for n in nu: 

                print(f"Testing {k}, {g}, nu={n}")

                svm_small = OneClassSVM(kernel = k, gamma=g, nu=n)
                svm_small.fit(X1)
                svm_large = OneClassSVM(kernel = k, gamma = g, nu = n)
                svm_large.fit(X2)

                small_pred = svm_small.predict(Z1_val)
                y_pred_small = (small_pred == -1).astype(int)
                large_pred = svm_large.predict(Z2_val)
                y_pred_large = (large_pred == -1).astype(int)

                small_precision = precision_score(w1_val, y_pred_small, zero_division=0)
                small_recall = recall_score(w1_val, y_pred_small)
                small_f1 = f1_score(w1_val, y_pred_small)
                small_f2 = fbeta_score(w1_val, y_pred_small, beta=2)
                small_accuracy = accuracy_score(w1_val, y_pred_small)
                roc_auc_small = roc_auc_score(w1_val, y_pred_small)

                results_small.append({
                    "kernel": k,
                    "gamma": g,
                    "nu": n,
                    "precision": small_precision,
                    "recall": small_recall,
                    "f1": small_f1,
                    "f2": small_f2,
                    "accuracy": small_accuracy,
                    "roc_auc": roc_auc_small
                })

                large_precision = precision_score(w2_val, y_pred_large, zero_division=0)
                large_recall = recall_score(w2_val, y_pred_large)
                large_f1 = f1_score(w2_val, y_pred_large)
                large_f2 = fbeta_score(w2_val, y_pred_large, beta=2)
                large_accuracy = accuracy_score(w2_val, y_pred_large)
                roc_auc_large = roc_auc_score(w2_val, y_pred_large)

                results_large.append({
                    "kernel": k,
                    "gamma": g,
                    "nu": n,
                    "precision": large_precision,
                    "recall": large_recall,
                    "f1": large_f1,
                    "f2": large_f2,
                    "accuracy": large_accuracy,
                    "roc_auc": roc_auc_large
                })

                if small_f2 > best_small:
                    best_small = small_f2
                    best_small_model = svm_small
                    best_params = {'kernel': k, 'gamma': g, 'nu': n}

                if large_f2 > best_large:
                    best_large = large_f2
                    best_large_model = svm_large
                    best_params_large = {'kernel': k, 'gamma': g, 'nu': n}

    df_small = pd.DataFrame(results_small)
    df_large = pd.DataFrame(results_large)

    df_small = df_small.sort_values("f2", ascending=False)
    df_large = df_large.sort_values("f2", ascending=False)

    df_small.to_csv(
        os.path.join(small_path, "grid_search_results.csv"),
        index=False
    )

    df_large.to_csv(
        os.path.join(large_path, "grid_search_results.csv"),
        index=False
    )

    print("\nBEST PARAMETERS small")
    print(best_params)
    print("BEST VAL F1:", best_small)

    print("\nBEST PARAMETERS large")
    print(best_params_large)
    print("BEST VAL F1:", best_large)

    os.makedirs(os.path.dirname(small_path), exist_ok=True)

    joblib.dump(
    best_small_model,
    os.path.join(small_path, "best_small_ocsvm.pkl")
)

    os.makedirs(os.path.dirname(large_path), exist_ok = True)

    joblib.dump(
    best_large_model,
    os.path.join(large_path, "best_large_ocsvm.pkl")
)

    # Save best small SVM parameters
    with open(os.path.join(small_path, "best_small_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)


    # Save best large SVM parameters
    with open(os.path.join(large_path, "best_large_params.json"), "w") as f:
        json.dump(best_params_large, f, indent=4)

    Z1, w1 = extract_features(small_model, test_loader, 'fc1', device)
    Z2, w2 = extract_features(large_model, test_loader, 'fc.0', device)

    small_pred = best_small_model.predict(Z1)
    y_pred_small = (small_pred == -1).astype(int)
    large_pred = best_large_model.predict(Z2)
    y_pred_large = (large_pred == -1).astype(int)

    roc_auc_small = roc_auc_score(w1, y_pred_small)
    roc_auc_large = roc_auc_score(w2, y_pred_large)

    cr_small = classification_report(w1, y_pred_small)
    cr_large = classification_report(w2, y_pred_large)

    print(f' small SVM Classification Report:\n{classification_report(w1, y_pred_small)}')
    print(f' small SVM ROC:\n{roc_auc_small}')
    print(f' large SVM Classification Report:\n{classification_report(w2, y_pred_large)}')
    print(f' large SVM ROC:\n{roc_auc_large}')

    from sklearn.metrics import confusion_matrix


    print("small:")
    print(confusion_matrix(w1, y_pred_small))

    print("large:")
    print(confusion_matrix(w2, y_pred_large))