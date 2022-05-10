import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.data.dataloader import SLAPDataModule, SLAPDataset
from src.model.mpnn_classifier import DMPNNModel
from src.util.configuration import CONFIG
from src.util.definitions import DATA_ROOT, PROJECT_DIR, LOG_DIR_ABS
from src.cross_validation import cross_validate
from src.hyperopt import optimize_hyperparameters
# logging
os.environ["WANDB_DIR"] = LOG_DIR_ABS
os.environ["WANDB_MODE"] = "online"
wandb_group_name = wandb.util.generate_id()
wandb_project = "slap-gnn"
# cross_validate = 5
log_cv_mean = True


# TODO: have the possibility to pass train and test set, then internally split train into train and val and do cv on it

# data
# dm = SLAPDataModule(data_dir=DATA_ROOT,
#                     split_dir=DATA_ROOT / CONFIG["split_dir_name"],
#                     cross_validate=cross_validate,
#                     reaction=CONFIG["reaction"],
#                     smiles_columns=("SMILES", ),
#                     label_column="targets",
#                     batch_size=32,
#                     )
# dm.prepare_data()
# dm.setup()

data = SLAPDataset(raw_dir=DATA_ROOT, reaction=True, smiles_columns=("SMILES", ), label_column="targets")
CONFIG["atom_feature_size"] = data.atom_feature_size
CONFIG["bond_feature_size"] = data.bond_feature_size
print(optimize_hyperparameters(CONFIG, data))
exit(0)
print(cross_validate(CONFIG, data, ))
exit(0)

# run training
if cross_validate:
    metrics = []
    for fold in range(cross_validate):
        wandb.init(group=wandb_group_name, project=wandb_project, name=f"fold{fold}")
        model = DMPNNModel(**CONFIG)
        # trainer
        logger = WandbLogger()
        trainer = pl.Trainer(max_epochs=CONFIG["training"]["max_epochs"], log_every_n_steps=1,
                             default_root_dir=PROJECT_DIR, logger=logger)
        trainer.fit(model, train_dataloaders=dm.train_dataloader(fold), val_dataloaders=dm.val_dataloader(fold))
        metrics.append(trainer.logged_metrics)
        wandb.log(trainer.logged_metrics)

        wandb.finish()
        trainer.logged_metrics.keys()
    if log_cv_mean:
        metrics_mean = {}
        for k in trainer.logged_metrics.keys():
            if not k.startswith("_"):
                metrics_mean[k] = torch.mean(torch.stack([fold[k] for fold in metrics]))
        wandb.init(group=wandb_group_name, project=wandb_project, name="CV_summary")
        wandb.log(metrics_mean)
        wandb.finish()
else:
    wandb.init(group=wandb_group_name, project=wandb_project)
    model = DMPNNModel(**CONFIG)
    # trainer
    logger = WandbLogger()
    trainer = pl.Trainer(max_epochs=CONFIG["training"]["max_epochs"], log_every_n_steps=1, default_root_dir=PROJECT_DIR,
                         logger=logger)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())
    wandb.finish()



