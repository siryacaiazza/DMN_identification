import pandas as pd
import os
import nibabel as nib

DATA_DIR = '/path/to/data'

def build_df(DATA_DIR):
    df = pd.DataFrame()
    for type in os.listdir(DATA_DIR):
        type_path = os.path.join(DATA_DIR, type)
        if os.path.isdir(type_path):
            for patient in os.listdir(type_path):
                ICA_path = os.path.join(type_path, patient, 'ICA.ica', 'filtered_func_data.ica', 'data_2channels', 'melodic_IC.npy')
                label_path = os.path.join(type_path, patient, 'ICA.ica', 'filtered_func_data.ica', 'label') 
                ni_path = os.path.join(type_path, patient, 'ICA.ica', 'filtered_func_data.ica', 'melodic_IC.nii.gz') 
                N_ICA = nib.load(ni_path).shape[-1] if os.path.exists(ni_path) else 0 
                if os.path.exists(ICA_path):
                    df = pd.concat([
    df,
    pd.DataFrame([{
        'type': type,
        'patient': patient,
        'ICA_path': ICA_path,
        'N_ICA': N_ICA,
        'label_path': label_path
    }])
], ignore_index=True)
    return df

df = build_df(DATA_DIR)
df.sort_values(by=['patient'], inplace=True)
# save the dataframe to a csv file
df.to_csv('dataframe.csv', index=False)