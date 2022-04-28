import torch
import pytorch_lightning as pl

from src.data.dataloader import SLAPDataModule
from src.model.mpnn_classifier import DMPNNModel
from src.util.configuration import CONFIG
from src.util.definitions import DATA_ROOT


dm = SLAPDataModule(data_dir=DATA_ROOT,
                    split_dir=DATA_ROOT / "example_split",
                    reaction=CONFIG["reaction"],
                    smiles_columns=("SMILES", ),
                    label_column="targets"
                    )

dm.prepare_data()
dm.setup()
dl_iter = iter(dm.train_dataloader())
batch = next(dl_iter)

model = DMPNNModel(**CONFIG, atom_feature_size=dm.atom_feature_size, bond_feature_size=dm.bond_feature_size)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)

trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

