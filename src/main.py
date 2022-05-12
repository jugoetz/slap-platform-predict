from src.data.dataloader import SLAPDataset
from src.util.configuration import CONFIG
from src.util.definitions import DATA_ROOT, PROJECT_DIR
from src.cross_validation import cross_validate
from src.train import run_training
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from  pytorch_lightning.loggers.wandb import WandbLogger
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
callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=5))

# initialize trainer
if CONFIG["gpu"] is True:
    accelerator = "gpu"
else:
    accelerator = "cpu"
trainer = pl.Trainer(max_epochs=CONFIG["training"]["max_epochs"], log_every_n_steps=1,
                     default_root_dir=PROJECT_DIR, logger=WandbLogger(), accelerator=accelerator, callbacks=callbacks)

# split data
train_idx = pd.read_csv(DATA_ROOT / CONFIG["split_dir_name"] / "fold_0_train_idx.csv", header=0).values.flatten()
test_idx = pd.read_csv(DATA_ROOT / CONFIG["split_dir_name"] / "fold_0_test_idx.csv", header=0).values.flatten()
train = torch.utils.data.Subset(data, train_idx)
test = torch.utils.data.Subset(data, test_idx)

# run training
metrics, model = run_training(CONFIG, train, trainer, save_model=True)

if run_test_set:

    # TODO configure checkpoint callback for trainer
    print(trainer.test(model, test, ckpt_path="best"))
    # fixme: there is a problem in test_step()




