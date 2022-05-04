import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.data.dataloader import SLAPDataModule
from src.model.mpnn_classifier import DMPNNModel
from src.util.configuration import CONFIG
from src.util.definitions import DATA_ROOT, PROJECT_DIR, LOG_DIR_ABS

# logging
os.environ["WANDB_DIR"] = LOG_DIR_ABS
os.environ["WANDB_MODE"] = "online"
wandb_group_name = "test_CV-4"
wandb_project = "slap-gnn"


# TODO: have the possibility to pass train and test set, then internally split train into train and val and do cv on it
cross_validate = 5
# data
dm = SLAPDataModule(data_dir=DATA_ROOT,
                    split_dir=DATA_ROOT / CONFIG["split_dir_name"],
                    cross_validate=cross_validate,
                    reaction=CONFIG["reaction"],
                    smiles_columns=("SMILES", ),
                    label_column="targets",
                    batch_size=32,
                    )
dm.prepare_data()
dm.setup()



# run training
if cross_validate:
    metrics = []
    for fold in range(cross_validate):
        wandb.init(group=wandb_group_name, project=wandb_project)
        model = DMPNNModel(**CONFIG, atom_feature_size=dm.atom_feature_size, bond_feature_size=dm.bond_feature_size)
        # trainer
        logger = WandbLogger()
        trainer = pl.Trainer(max_epochs=CONFIG["training"]["max_epochs"], log_every_n_steps=1,
                             default_root_dir=PROJECT_DIR, logger=logger)
        trainer.fit(model, train_dataloaders=dm.train_dataloader(fold), val_dataloaders=dm.val_dataloader(fold))
        metrics.append(trainer.logged_metrics)
        wandb.log(trainer.logged_metrics)
        wandb.join()
    print([fold["val/accuracy"] for fold in metrics])

else:
    wandb.init(group=wandb_group_name, project=wandb_project)
    model = DMPNNModel(**CONFIG, atom_feature_size=dm.atom_feature_size, bond_feature_size=dm.bond_feature_size)
    # trainer
    logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=CONFIG["training"]["max_epochs"], log_every_n_steps=1, default_root_dir=PROJECT_DIR,
                         logger=logger)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    wandb.join()
