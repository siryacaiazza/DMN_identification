import os
from random import sample
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader 
import nibabel as nib
import random
from pathlib import Path

# TEMPLATE_PATH = '/home/siryacaiazza/fsl/data/standard/MNI152_T1_2mm.nii.gz'

class FMRIDataset(Dataset):
    """
    Dataset class for fMRI ICA components classification.
    Loads 3D ICA components and their labels.
    """
    def __init__(self, csv_path):
        """
        Args:
            csv_path: Path to dataframe.csv containing image paths
        """
        self.csv_path = csv_path
        self.data = self.load_data()

    
    def load_data(self):
        """Load CSV and create list of samples"""
        df = pd.read_csv(self.csv_path)
        samples = []
        
        for _, row in df.iterrows():
            ica_path = row['ICA_path']
            label_path = row['label_path']
            patient_type = row['type']
            patient = row['patient']
            n_ica = int(row['N_ICA'])
            
            # Verify paths exist
            if not os.path.exists(ica_path):
                print(f"Warning: ICA path not found: {ica_path}")
                continue
            if not os.path.exists(label_path):
                print(f"Warning: Label path not found: {label_path}")
                continue

            comp_labels = self._load_component_labels(label_path)
            for comp_idx in range(n_ica):
                label = comp_labels.get(comp_idx + 1, 0)  # component indices start at 1
                samples.append({
                        'ica_path': ica_path,
                        'component_idx': comp_idx,
                        'label': label,
                        'patient_type': patient_type,
                        'patient': patient
                    })
        

        return samples
    
    def _load_component_labels(self, label_path):
        """
        Load component labels from file.
        File format: component_id, label
        Returns dict mapping component_id 1 if DMN, 0 if not DMN
        """
        comp_labels = {}
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()

                for line in lines[:-1]:   # skips the last line
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) == 3:
                        comp_id = int(parts[0])
                        if parts[1] == 'DMN':
                            comp_labels[comp_id] = 1 
                        else:
                            comp_labels[comp_id] = 0
        except Exception as e:
            print(f"Error loading labels from {label_path}: {e}")

        net_mapping = {
            'DMN': 1,
            'Not DMN': 0
        }
        
        return comp_labels
    
    def _encode_patient_type(self, patient_type):
        """Encode patient type as integer label"""
        type_mapping = {
            'Epilessia': 0,
            'Neurochirurgia': 1,
            'Neurorianimazione': 2,
            'Volontari': 3
        }
        return type_mapping.get(patient_type, -1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        sample = self.data[idx] 

        # ICA cache
        base = str(Path(sample['ica_path']).with_suffix(''))
        comp_path = f"{base}_comp{sample['component_idx']}.npy"
        img_data = np.load(comp_path)

           
        # Convert to tensor and add channel dimension for CNN
        img_tensor = torch.from_numpy(np.array(img_data, dtype=np.float32))
        label = torch.tensor(sample['label'], dtype=torch.long) 
        
        return img_tensor, label


from sklearn.model_selection import train_test_split

class DataLoader3D:
    @staticmethod
    def create_loaders(csv_path, batch_size=16, num_workers=4):

        dataset = FMRIDataset(csv_path)

        # Build patient-level table
        patient_to_type = {}
        for sample in dataset.data:
            patient_to_type[sample['patient']] = sample['patient_type']

        patients = np.array(list(patient_to_type.keys()))
        patient_types = np.array([patient_to_type[p] for p in patients])

        # Train/Test split (patient-level)
 

        train_patients, test_patients = train_test_split(
            patients,
            test_size=0.2,
            random_state=42,
            stratify=patient_types
        )

        # rebuild mapping for train split
        train_types = np.array([patient_to_type[p] for p in train_patients])

        train_patients, val_patients = train_test_split(
            train_patients,
            test_size=0.25,  # 0.25 of train = 0.2 of total
            random_state=42,
            stratify=train_types
        )

        train_patients = set(train_patients)
        val_patients = set(val_patients)
        test_patients = set(test_patients)

        # Map back to indices

        train_indices, val_indices, test_indices = [], [], []

        for i, sample in enumerate(dataset.data):
            p = sample['patient']
            if p in train_patients:
                train_indices.append(i)
            elif p in val_patients:
                val_indices.append(i)
            elif p in test_patients:
                test_indices.append(i)


        # Create subsets

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        return train_loader, val_loader, test_loader


# Example usage:
if __name__ == "__main__":
    csv_path = 'path/to/.csv'
    
    # Create dataset for component classification
    dataset = FMRIDataset(csv_path)
    
    # Get one sample
    if len(dataset) > 0:
        sample_tensor, struct_tensor, label = dataset[0]
        print(f"Sample shape: {sample_tensor.shape}") 
        print(f"Label: {label}")
    
    # Label distribution in the dataset

    labels = [sample['label'] for sample in dataset.data]
    label_counts = pd.Series(labels).value_counts()
    print(f"Label distribution: {label_counts.to_dict()}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = DataLoader3D.create_loaders(csv_path, batch_size=1)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    print(f"Total samples: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}")  
    print(f"Train batch shape: {next(iter(train_loader))[0].shape}") 
