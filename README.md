## Installation
```bash
conda env create -f environment.yaml
```

---

## Usage
```bash
python run.py --config <path_to_config_file>
```
Will run train and evaluate a model under cross-validation.

Optional flags:
- ```--hparam_optimization```: Run Bayesian optimization of hyperparameters
  - ```--hparam_config_path <path_to_file>```: Config file for hyperparameter optimization bounds
  - ```--hparam_n_iter <n>```: Number of iterations for hyperparameter optimization

### Data
Data sets are read from `$PROJECT_ROOT/data`. The data sets must be CSV files with one column `SMILES` and one column `targets`.

Splits for the data sets have to be provided as five separate CSV files per fold, e.g.,
`fold0_val.csv`, `fold0_test_0D.csv` (currently, these names are hardcoded into `run_experiment.py`).
Each of these files contains one column of indices for the respective split with no header.

The file name and directory for the data set and the split directory can be set in the config file.

### Model configuration
Edit configuration in the config file passed to `--config`.
