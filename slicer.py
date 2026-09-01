import numpy as np
import pandas as pd
import os
from pathlib import Path

csv_path = 'path/to/.csv'
df = pd.read_csv(csv_path)

unique_ica_paths = df['ICA_path'].unique()
print(f"File ICA da convertire: {len(unique_ica_paths)}")

for i, ica_path in enumerate(unique_ica_paths):
    base = str(Path(ica_path).with_suffix(''))  # rimuove .npy
    
    # Controlla se già convertito (es. se riesegui lo script dopo un'interruzione)
    first_comp_path = f"{base}_comp0.npy"
    if os.path.exists(first_comp_path):
        print(f"[{i+1}/{len(unique_ica_paths)}] Già convertito, skip: {ica_path}")
        continue
    
    print(f"[{i+1}/{len(unique_ica_paths)}] Carico: {ica_path}")
    data = np.load(ica_path)  # (2, 80, 80, 56, n_ica) — unico carico pesante
    n_ica = data.shape[-1]
    
    for comp_idx in range(n_ica):
        out_path = f"{base}_comp{comp_idx}.npy"
        np.save(out_path, data[..., comp_idx])
    
    del data  # libera RAM subito
    print(f"  -> Salvate {n_ica} componenti")

print("Conversione completata.")