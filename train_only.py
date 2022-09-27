"""Quick solution to train a model with known set of hyperparameters"""
import argparse
import os

import numpy as np
import wandb
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.data.dataloader import SLAPDataset, collate_fn
from src.model.classifier import DMPNNModel, GCNModel, FFNModel
from src.util.configuration import get_config
from src.util.definitions import LOG_DIR, DATA_ROOT, PROJECT_DIR
from src.util.logging import generate_run_id

os.environ['WANDB_MODE'] = 'offline'


def train(data, hparams, trainer, run_id=None, save_model=True):
    """
    Trains a model on given data with one set of hyperparameters.

    Args:
        data (torch.utils.data.DataLoader): Training data
        hparams (dict): Model hyperparameters
        run_id (optional, str): Unique id to identify the run. If None, will generate an ID containing the current datetime.
        save_model (bool): Whether to save the trained model weights to disk. Defaults to False.

    Returns:
        dict: Dictionary of validation metrics and, if return_test_metrics is True, additionally test metrics
        DMPNNModel: Trained model
    """
    # generate run_id if None is passed
    if not run_id:
        run_id = generate_run_id()

    wandb.init(reinit=True, project="slap-gnn", name=run_id, group="single_point_training", config=hparams)

    # initialize model
    if hparams["encoder"]["type"] == "D-MPNN":
        model = DMPNNModel(**hparams)
    elif hparams["encoder"]["type"] == "GCN":
        model = GCNModel(**hparams)
    else:
        model = FFNModel(**hparams)

    # run training
    trainer.fit(model, train_dataloaders=data)
    # optionally, save model weights
    if save_model:
        trainer.save_checkpoint(filepath=LOG_DIR / run_id / "model_checkpoints", weights_only=True)

    wandb.finish()
    return model


def predict(model, data, trainer):
    """
    Predicts on given data with given model.

    Args:
        model (DMPNNModel): Trained model
        data (torch.utils.data.DataLoader): Data to predict on

    Returns:
        dict: Dictionary of predictions
    """
    probs = torch.cat(trainer.predict(model, dataloaders=data))
    # obtain labels from probabilities at 0.5 threshold
    labels = (probs > 0.5).int()
    return labels


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
args = parser.parse_args()

config = get_config(args.config)
# load data
data = SLAPDataset(name=config["data_name"],
                   raw_dir=DATA_ROOT,
                   reaction=config["reaction"],
                   smiles_columns=("SMILES", ),
                   label_column="targets",
                   graph_type=config["graph_type"],
                   global_features=config["global_features"],
                   featurizers=config["featurizers"],
                   )

dl = DataLoader(data, batch_size=32, shuffle=True, collate_fn=collate_fn)

# update config with data processing specifics
config["atom_feature_size"] = data.atom_feature_size
config["bond_feature_size"] = data.bond_feature_size
config["global_feature_size"] = data.global_feature_size
trainer = pl.Trainer(max_epochs=config["training"]["max_epochs"], log_every_n_steps=1,
                                  default_root_dir=PROJECT_DIR,
                                  accelerator=config["accelerator"])
model = train(dl, config, trainer)


predict_data = SLAPDataset(name="validation_plate_featurized_partial.csv",
                   raw_dir=DATA_ROOT,
                   reaction=config["reaction"],
                   smiles_columns=("reactionSMILES", ),
                   label_column=None,
                   graph_type=config["graph_type"],
                   global_features=config["global_features"],
                   featurizers=config["featurizers"],
                   )


predict_dl = DataLoader(predict_data, batch_size=32, collate_fn=collate_fn, shuffle=False)

predictions = predict(model, predict_dl, trainer)
# save predictions to text file
with open("predictions.txt", "w") as f:
    for i in predictions.tolist():
        f.write(str(i) + "\n")

print(predictions)
