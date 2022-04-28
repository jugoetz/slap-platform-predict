import pathlib

import torch
import pytorch_lightning as pl

from src.data.dataloader import SLAPDataModule
from src.model.mpnn_classifier import DMPNNModel

# todo this is just a debug configuration, and it should be read from file
hyperparams = {
    "lr": 1e-4,
    "weight_decay": 0.001,  # todo: no idea what to set this to
    "lr_scheduler": {
        "scheduler_name": "exp_with_linear_warmup",
        "lr_warmup_step": 5,
        "lr_min": 1e-5,
        "epochs": 100,
    },
    "reaction": True,
               }

data_root = pathlib.Path(__file__).parent.parent / "data"
print(data_root)

dm = SLAPDataModule(data_dir=data_root,
                    split_dir=data_root / "example_split",
                    reaction=hyperparams["reaction"],
                    smiles_columns=("SMILES", ),
                    label_column="targets"
                    )

dm.prepare_data()
dm.setup()
dl_iter = iter(dm.train_dataloader())
batch = next(dl_iter)



model = DMPNNModel(**hyperparams)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)

trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

