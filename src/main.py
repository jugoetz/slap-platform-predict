from src.data.dataloader import SLAPDataset
from src.util.configuration import CONFIG
from src.util.definitions import DATA_ROOT
from src.cross_validation import cross_validate


# load data
data = SLAPDataset(name=CONFIG["data_name"], raw_dir=DATA_ROOT, reaction=CONFIG["reaction"], smiles_columns=("SMILES", ), label_column="targets")

# update config with data processing specifics
CONFIG["atom_feature_size"] = data.atom_feature_size
CONFIG["bond_feature_size"] = data.bond_feature_size

# TODO split data



# run training
aggregate_metrics, fold_metrics = cross_validate(CONFIG, data, 1, save_models=True, return_fold_metrics=True)




