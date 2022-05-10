from src.data.dataloader import SLAPDataset
from src.util.configuration import CONFIG
from src.util.definitions import DATA_ROOT
from src.cross_validation import cross_validate


# cross_validate = 5
log_cv_mean = True

# load data
data = SLAPDataset(raw_dir=DATA_ROOT, reaction=True, smiles_columns=("SMILES", ), label_column="targets")

# update config with data processing specifics
CONFIG["atom_feature_size"] = data.atom_feature_size
CONFIG["bond_feature_size"] = data.bond_feature_size


# run training
print(cross_validate(CONFIG, data, 1, save_models=True))




