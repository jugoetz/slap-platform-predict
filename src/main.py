import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.data.dataloader import SLAPDataModule
from src.model.mpnn_classifier import DMPNNModel
from src.util.configuration import CONFIG
from src.util.definitions import DATA_ROOT, PROJECT_DIR, LOG_DIR_ABS

# logging
os.environ["WANDB_DIR"] = LOG_DIR_ABS
logger = WandbLogger(project="slap-gnn", offline=True)

# data
dm = SLAPDataModule(data_dir=DATA_ROOT,
                    split_dir=DATA_ROOT / CONFIG["split_dir_name"],
                    reaction=CONFIG["reaction"],
                    smiles_columns=("SMILES", ),
                    label_column="targets",
                    batch_size=32,
                    )
dm.prepare_data()
dm.setup()

# model
model = DMPNNModel(**CONFIG, atom_feature_size=dm.atom_feature_size, bond_feature_size=dm.bond_feature_size)

# trainer
trainer = pl.Trainer(max_epochs=10, log_every_n_steps=1, default_root_dir=PROJECT_DIR, logger=logger)

# run training
trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
