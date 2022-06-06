from copy import deepcopy

from src.data.dataloader import SLAPDataset
from src.util.configuration import CONFIG
from src.util.definitions import DATA_ROOT, PROJECT_DIR
from src.cross_validation import cross_validate, cross_validate_predefined
from src.hyperopt import optimize_hyperparameters_bayes
from src.train import run_training


run_test_set = True

# load data
data = SLAPDataset(name=CONFIG["data_name"], raw_dir=DATA_ROOT, reaction=CONFIG["reaction"], smiles_columns=("SMILES", ), label_column="targets")

# update config with data processing specifics
CONFIG["atom_feature_size"] = data.atom_feature_size
CONFIG["bond_feature_size"] = data.bond_feature_size

# initialize callbacks
callbacks = []
# callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=5))


# define split index files
split_files = [{"train": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_train.csv",
                "val": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_val.csv",
                "test_0D": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_test_0D.csv",
                "test_1D": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_test_1D.csv",
                "test_2D": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_test_2D.csv"}
               for i in range(5)]



# let's search a bunch of hparams we are interested in
# lrs = [1e-4, 3e-3, 1e-3, 3e-3, 1e-2]
# enc_hiddensizes = [32, 64, 128, 256, 512, 1028, 2048]
# enc_depths = [3, 4, 5, 6]
# dec_hiddensizes = [16, 32, 64, 128, 256]
# dec_depths = [1, 2, 3]
# global_dropouts = [0., 0.05, 0.1, 0.2, 0.4, 0.6]
# aggregation = ["attention", "mean", "max", "sum"]


# run cross-validation with configured hparams
# aggregate_metrics, fold_metrics = cross_validate_predefined(CONFIG, data, split_files=split_files, save_models=False, return_fold_metrics=True)
# print(aggregate_metrics)
# print(fold_metrics)

# or run bayesian hparam optimization
best_params, values, experiment = optimize_hyperparameters_bayes(CONFIG, data, split_files)
print(best_params, values)
