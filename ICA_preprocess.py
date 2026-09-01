import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import pandas as pd

csv_path = 'path/to/dataframe.csv'
df = pd.read_csv(csv_path)

def preprocess_ica(ica: np.ndarray) -> np.ndarray:
    """
    Restituisce un array (2, X, Y, Z) con i canali:
      0 - z-score positivo 
      1 - z-score negativo 
    """
    # --- maschera cervello ---
    brain_mask = (np.abs(ica) > 0).astype(np.float32)

    # --- z-score SOLO sui voxel cerebrali ---

    z   = ica * brain_mask          # azzera fuori dal cervello

    # --- canali positivo / negativo con clipping a ±4σ ---
    slices = []
    for i in range(ica.shape[-1]):
        pos = np.maximum(z[...,i],  0) 
        neg = np.maximum(-z[...,i], 0) 
        pos = pos / (np.percentile(pos, 99) + 1e-6)
        neg = neg / (np.percentile(neg, 99) + 1e-6)

        volume = np.stack([pos,neg], axis=0)

        slices.append(volume)

    # --- stack canali ---
    slices = np.stack(slices, axis=0)  # (3, X, Y, Z, N_ica)
    slices = np.moveaxis(slices, 0, -1)

    return slices.astype(np.float32)


for _, row in tqdm(df.iterrows(), total=len(df)):

    ica_path = row['ICA_path']
    if not os.path.exists(ica_path):
        print(f"Missing ICA: {ica_path}")
        continue

    out_dir = os.path.join(os.path.dirname(ica_path), "data_2channels")
    os.makedirs(out_dir, exist_ok=True)

    filename = os.path.basename(ica_path).replace(".nii.gz", ".npy")
    new_path = os.path.join(out_dir, filename)

    
    ica = nib.load(ica_path).get_fdata().astype(np.float32)
    print(f'ica shape {ica.shape}')
    volume = preprocess_ica(ica)
    print(f'volume shape {volume.shape}')
    np.save(new_path, volume)