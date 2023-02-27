## Installation
```bash
conda env create -f environment.yaml
```

---
## Usage (Use our trained models to make a prediction)

You will need a CSV-file containing a column `smiles` containing the SMILES strings of your query molecules.
The CSV-file can contain arbitrary additional columns, one containing an identifier is recommended.

To make a prediction, run the following command:
```bash
python inference.py --product-file PRODUCT_FILE
```
where `PRODUCT_FILE` is the path to the CSV-file containing the query molecules.
When passing the `--reaction` flag, reactionSMILES strings are expected as input instead of SMILES strings of the product.

Alternatively, you can use the jupyter notebook `notebooks/inference.ipynb` for more flexibility,
but the command line interface should fit most needs.


## Usage (Training/inference on new data)
```bash
# Train and evaluate a model under cross-validation:
python run.py train --config CONFIG --data_path DATA_PATH

# For information on optional arguments use:
python run.py train -h
```


```bash
# Make predictions using a trained model:
python run.py predict [-h] --config CONFIG --data_path DATA_PATH --model_path MODEL_PATH

# For information on optional arguments use:
python run.py predict -h
```


### Data
Data sets are read from `DATA_PATH`. The data sets must be CSV files with one column `SMILES` and one column `targets`.
Depending on the model configuration the `SMILES` column should contain SMILES strings of the intermediate or
atom-mapped, unbalanced reactionSMILES strings of the reaction. In the context of the paper, the latter option is used.
The `targets` column should contain value `0` or `1` for unsuccessful and successful reactions, respectively.

Splits for the data sets can be provided as five separate CSV files per fold, e.g.,
`fold0_val.csv`, `fold0_test_0D.csv` if the `predefined` split strategy is used.
Each of these files contains one column of indices for the respective split with no header.
Other strategies (`random` and `KFold`) do not require split index files.


### Model configuration
Edit configuration in the .yaml-file passed to `--config`.
A sample can be found in `config/`.
Parameters given in the config file are overwritten by command-line arguments.


### Hyperparameter search configuration
Hyperparameter searches require an additional configuration file containing the hyperparameter grid.
The file is passed to `--hparam_config_path`
For an example, refer to `config/hparam_bounds.yaml`.
