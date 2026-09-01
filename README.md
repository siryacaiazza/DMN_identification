# DMN Identification from fMRI

This repository contains the code for the identification of **Default Mode Network (DMN)** components from fMRI data using machine learning and deep learning approaches.

The project focuses on the classification of ICA-derived fMRI components as either **DMN** or **non-DMN**. Two complementary approaches are explored:

- **3D Convolutional Neural Networks (CNNs)** operating directly on volumetric ICA components.
- **Support Vector Machine (SVM)**-based models for component classification.

The repository also includes scripts for dataset construction, ICA preprocessing, data loading, and conversion of multi-component volumes into individual samples.

---

## Project Overview

The general processing pipeline is:

```text
fMRI data
   │
   ▼
ICA decomposition
   │
   ▼
ICA components (.nii.gz)
   │
   ▼
Preprocessing
   │
   ├── Positive / negative channels
   └── Normalization
   │
   ▼
Preprocessed components (.npy)
   │
   ▼
Dataset construction
   │
   ▼
Train / Validation / Test split
   │
   ├───────────────┐
   ▼               ▼
  3D CNN          SVM
   │               │
   └───────┬───────┘
           ▼
     DMN identification
```

The preprocessing pipeline converts ICA components into a two-channel representation containing the positive and negative portions of each component. The resulting volumes are stored as NumPy arrays and used as input for the classification models.

---

## Repository Structure

```text
DMN_identification/
│
├── CNN/
│   ├── CNN_training.py
│   └── CNN_testing.py
│
├── SVM/
│   ├── SVM.py
│   ├── best_large_ocsvm.pkl
│   ├── best_large_params.json
│   ├── best_small_ocsvm.pkl
│   ├── best_small_params.json
│   └── large_siamese_history.dict
│
├── ICA_preprocess.py
├── dataframe_builder.py
├── dataloader.py
├── slicer.py
└── requirements.txt
```

### Main scripts

#### `dataframe_builder.py`

Builds a dataframe containing the information required to access the dataset.

For each subject, the dataframe stores information such as:

- subject type
- patient identifier
- ICA component path
- number of ICA components
- component label path

The resulting dataframe is saved as:

```text
dataframe.csv
```



#### `ICA_preprocess.py`

Preprocesses the ICA components stored as NIfTI files.

The preprocessing creates a two-channel representation:

1. Positive component values
2. Negative component values

The processed data are stored as NumPy arrays (`.npy`).

#### `slicer.py`

Splits the preprocessed multi-component ICA volume into individual component files.

For example:

```text
melodic_IC.npy
```

is converted into individual components such as:

```text
melodic_IC_comp0.npy
melodic_IC_comp1.npy
melodic_IC_comp2.npy
...
```

This allows individual ICA components to be used as samples during model training.

#### `dataloader.py`

Provides the PyTorch dataset and data-loading utilities used by the CNN models.

The `FMRIDataset` class loads ICA components together with their corresponding labels and metadata.

---

## CNN Approach

The `CNN/` directory contains the training and testing code for the convolutional neural network approach.

### Training

`CNN_training.py` trains two 3D CNN configurations:

- **Small CNN**
- **Large CNN**

The models operate on 3D fMRI ICA component volumes and perform binary classification:

```text
0 → Not DMN
1 → DMN
```

The training pipeline uses:

- PyTorch
- Adam optimizer
- weighted cross-entropy loss
- learning-rate scheduling
- early stopping

Because the dataset is highly imbalanced, class weights are used during training. The training code also evaluates the models using metrics including:

- F1 score
- ROC-AUC
- confusion matrix
- classification report



### Testing

`CNN_testing.py` contains the testing/evaluation pipeline for the trained CNN models.

---

## SVM Approach

The `SVM/` directory contains the SVM-based approach for DMN identification.

The repository includes trained model artifacts and parameter files for different configurations, including small and large models.

This approach provides an alternative to the end-to-end 3D CNN classification pipeline.

---

## Requirements

The project is implemented in Python and relies on the following packages:

```text
pandas
nibabel
nilearn
matplotlib
scipy
scikit-learn
seaborn
keras
optuna
torch
```

The complete list is available in `requirements.txt`.

### Installation

Clone the repository:

```bash
git clone https://github.com/siryacaiazza/DMN_identification.git
cd DMN_identification
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

The project expects fMRI data that have already undergone ICA decomposition.

The directory structure used by the preprocessing scripts follows a subject/type organization similar to:

```text
DATA_DIR/
├── type_1/
│   ├── patient_001/
│   │   └── ICA.ica/
│   │       └── filtered_func_data.ica/
│   │           ├── melodic_IC.nii.gz
│   │           └── label
│   │
│   └── patient_002/
│       └── ...
│
└── type_2/
    └── ...
```

The exact paths must be adapted to the local dataset.

> **Note:** The repository currently contains placeholder paths such as `path/to/dataframe.csv` and `path/to/data`. These must be replaced with the paths corresponding to your local environment before running the scripts.

---

## Preprocessing Pipeline

### 1. Build the dataframe

Set the dataset path in:

```python
DATA_DIR = '/path/to/data'
```

and run:

```bash
python dataframe_builder.py
```

This generates:

```text
dataframe.csv
```

containing the paths and metadata required by the subsequent steps.

### 2. Preprocess ICA components

Configure the dataframe path in `ICA_preprocess.py` and run:

```bash
python ICA_preprocess.py
```

The script converts the ICA NIfTI volumes into a normalized two-channel NumPy representation.

### 3. Slice individual components

After preprocessing, individual ICA components can be extracted using:

```bash
python slicer.py
```

The resulting component files can then be used by the data loader and classification models.

---

## Training the CNN

Before training, configure the paths in:

```text
CNN/CNN_training.py
```

The script creates the dataset loaders and trains both the small and large CNN models.

Run:

```bash
python CNN/CNN_training.py
```

The training procedure automatically:

1. Loads the dataset.
2. Creates training, validation and test loaders.
3. Trains the small CNN.
4. Trains the large CNN.
5. Evaluates validation performance.
6. Saves the best-performing model.
7. Stores the training history.

The current implementation uses CUDA automatically when a compatible GPU is available:

```python
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
```



---

## Model Evaluation

After training, the CNN models can be evaluated using:

```bash
python CNN/CNN_testing.py
```

The evaluation pipeline can be used to assess the ability of the trained models to distinguish DMN from non-DMN ICA components.

Relevant evaluation metrics include:

- Accuracy
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

---

## Class Imbalance

The dataset contains a strong imbalance between the two classes. The CNN training pipeline addresses this issue using weighted cross-entropy.

The class weights are computed from the number of samples in each class and passed to:

```python
torch.nn.CrossEntropyLoss
```

This gives greater importance to the minority DMN class during optimization.

---

## Reproducibility

To reproduce the experiments:

1. Prepare the fMRI/ICA dataset.
2. Configure the dataset paths.
3. Install the required Python dependencies.
4. Generate `dataframe.csv`.
5. Run the ICA preprocessing.
6. Split the ICA volumes into individual components.
7. Configure the CNN/SVM scripts.
8. Train the desired model.
9. Evaluate the model on the test set.

Because the current scripts contain environment-specific paths, these paths should be replaced before running the pipeline on a different machine.

---

## Limitations

This repository is currently a research-oriented implementation rather than a packaged software library.

Some configuration values, dataset paths and output directories are hard-coded in the scripts and therefore need to be adapted to the user's environment.

The dataset itself is not included in this repository.

---

## Future Work

Possible improvements include:

- [ ] Move dataset paths to a configuration file.
- [ ] Add command-line arguments for training and testing.
- [ ] Improve reproducibility through explicit random seeds.
- [ ] Add automated train/validation/test splitting.
- [ ] Add experiment configuration files.
- [ ] Provide pretrained model download instructions.
- [ ] Add visualization of identified DMN components.
- [ ] Add quantitative comparison between CNN and SVM approaches.
- [ ] Add automated experiment logging.

---

## Citation

If you use this repository in academic work, please cite the associated project or publication.

```text
Citation information will be added here.
```

---

## License

No license is currently specified for this repository.

If you intend to make the project openly reusable, consider adding an appropriate open-source license.