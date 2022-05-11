from src.data.dataloader import SLAPDataset
from src.util.configuration import CONFIG
from src.util.definitions import DATA_ROOT
from src.cross_validation import cross_validate

import torch
import pandas as pd

run_test_set = False

# load data
data = SLAPDataset(name=CONFIG["data_name"], raw_dir=DATA_ROOT, reaction=CONFIG["reaction"], smiles_columns=("SMILES", ), label_column="targets")

# update config with data processing specifics
CONFIG["atom_feature_size"] = data.atom_feature_size
CONFIG["bond_feature_size"] = data.bond_feature_size

# TODO split data
train_idx = pd.read_csv(DATA_ROOT / CONFIG["split_dir_name"] / "fold_0_train_idx.csv", header=0).values.flatten()
test_idx = pd.read_csv(DATA_ROOT / CONFIG["split_dir_name"] / "fold_0_test_idx.csv", header=0).values.flatten()
train = torch.utils.data.Subset(data, train_idx)
test = torch.utils.data.Subset(data, test_idx)

# run training
aggregate_metrics, fold_metrics = cross_validate(CONFIG, train, 1, save_models=True, return_fold_metrics=True)

if run_test_set:
    # TODO evaluate (need to return the trained model from cv)
    ...




