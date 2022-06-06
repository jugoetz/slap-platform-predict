from src.data.dataloader import SLAPDataset
from src.util.configuration import CONFIG
from src.util.definitions import DATA_ROOT, PROJECT_DIR
from src.cross_validation import cross_validate, cross_validate_predefined
from src.hyperopt import optimize_hyperparameters
from src.train import run_training
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
import pandas as pd

run_test_set = True

# load data
data = SLAPDataset(name=CONFIG["data_name"], raw_dir=DATA_ROOT, reaction=CONFIG["reaction"], smiles_columns=("SMILES", ), label_column="targets")

# update config with data processing specifics
CONFIG["atom_feature_size"] = data.atom_feature_size
CONFIG["bond_feature_size"] = data.bond_feature_size

# initialize callbacks
callbacks = []
# callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=5))

# initialize trainer
# trainer = pl.Trainer(max_epochs=CONFIG["training"]["max_epochs"], log_every_n_steps=1,
#            default_root_dir=PROJECT_DIR, logger=WandbLogger(), accelerator=config["accelerator"], callbacks=callbacks)

# define split index files
split_files = [{"train": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_train.csv",
                "val": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_val.csv",
                "test_0D": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_test_0D.csv",
                "test_1D": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_test_1D.csv",
                "test_2D": DATA_ROOT / "LCMS_split_763records" / f"fold{i}_test_2D.csv"}
               for i in range(5)]

# run cross-validation with configured hparams
# aggregate_metrics, fold_metrics = cross_validate_predefined(CONFIG, data, split_files=split_files, save_models=False, return_fold_metrics=True)
# print(aggregate_metrics)
# print(fold_metrics)

# or run hparam optimization
best_params, values, experiment = optimize_hyperparameters(CONFIG, data, split_files)
print(best_params, values)





