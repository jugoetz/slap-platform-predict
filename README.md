## Installation
```bash
conda env create -f environment.yaml
```

---

## Usage

### Data
Data sets are read from `$PROJECT_ROOT/data`. The data sets must be CSV files with one column `SMILES` and one column `targets`.

Splits for the data sets have to be provided as three separate CSV files `train_idx.csv`, `val_idx.csv`, and `test_idx.csv`.
Each of these files contains one column of indices for the respective split.

The file name and directory for the data set and the split directory can be set in `config/config.yaml`.

### Model configuration
Edit configuration in `config/config.yaml`.

### Train
```bash
python main.py
```

### Predict

to be added...