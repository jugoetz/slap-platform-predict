# SLAP Platform - Reaction Outcome Prediction
[![DOI:10.1126/sciadv.adj2314](https://zenodo.org/badge/DOI/10.1126/sciadv.adj2314.svg)](https://doi.org/10.1126/sciadv.adj2314)

Code to accompany the paper 

>J. Götz, M. K. Jackl, C. Jindakun, A. N. Marziale, J. André, D. J. Gosling, C. Springer, M. Palmieri, M. Reck, A. Luneau, C. E. Brocklehurst, J. W. Bode, High-throughput synthesis provides data for predicting molecular properties and reaction success *Sci. Adv.* **2023**, *9*, eadj2314.

See also:
[https://github.com/jugoetz/slap-platform](https://github.com/jugoetz/slap-platform) for the code used for data processing

## Installation

Install the required packages through conda. An `environment.yaml` file is provided to facilitate installation.
If your system does not support CUDA, use the `environment_cpuonly.yaml` file instead.
Installation will fail on M1/M2 Macs due to the `python=3.7` constraint. Remove it at your own risk.
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
where `PRODUCT_FILE` is the path to the CSV file containing the query molecules. The CSV file may be compressed in one
of the formats pandas can read (.gz, .bz2, .zip, .xz, .zst, .tar, .tar.gz, .tar.xz or .tar.bz2).
When passing the `--reaction` flag, atom-mapped reactionSMILES strings are expected as input instead of SMILES strings of the product.
If the flag is not passed, all atom-mapped reactionSMILES strings leading to the query molecule through the SLAP platform are generated and the model is applied to each of them.

Alternatively, you can use the jupyter notebook `notebooks/inference.ipynb` for more flexibility,
but the command line interface should fit most needs.


## Usage (Training/inference on new data)
```bash
# Train and evaluate a model under cross-validation:
python run.py train --config CONFIG --data-path DATA_PATH

# For information on optional arguments use:
python run.py train -h
```


```bash
# Make predictions using a trained model:
python run.py predict [-h] --config CONFIG --data-path DATA_PATH --model-path MODEL_PATH

# For information on optional arguments use:
python run.py predict -h
```


### Data
Data sets are read from `DATA_PATH`. The data sets must be CSV files with one column `SMILES` and one column `targets`.
Depending on the model configuration the `SMILES` column should contain SMILES strings of the intermediate or
atom-mapped, unbalanced reactionSMILES strings of the reaction. In the context of the paper, the latter option is used.
The `targets` column should contain value `0` or `1` for unsuccessful and successful reactions, respectively.
To train on our data sets, download the data from [Zenodo](https://doi.org/10.5281/zenodo.7950706).

Splits for the data sets can be provided as five separate CSV files per fold, e.g.,
`fold0_val.csv`, `fold0_test_0D.csv` if the `predefined` split strategy is used.
Each of these files contains one column of indices for the respective split with no header.
Other strategies (`random` and `KFold`) do not require split index files.


### Model configuration
Edit configuration in the .yaml-file passed to `--config`.
For an example, refer to `config/config.yaml`.
Parameters given in the config file are overwritten by command-line arguments.


### Hyperparameter search configuration
Hyperparameter searches require an additional configuration file containing the hyperparameter grid.
The file is passed to `--hparam-config-path`
For an example, refer to `config/hparam_bounds.yaml`.
