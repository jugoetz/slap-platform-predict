"""
Train a single model on a single fold with known set of hyperparameters.
Predict for a single test set.

Note: This was a quick implementation that is now going to be turned into a permanent script.

"""
import argparse
import os

import numpy as np
import wandb
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from src.data.dataloader import SLAPDataset, collate_fn
from src.model.classifier import Classifier, load_model
from src.util.configuration import get_config
from src.util.definitions import LOG_DIR, DATA_ROOT, PROJECT_DIR
from src.util.logging import generate_run_id

os.environ["WANDB_MODE"] = "offline"


def train(train_dl, val_dl, hparams, trainer, run_id=None, save_model=True):
    """
    Trains a model on given data with one set of hyperparameters.

    Args:
        train_dl (torch.utils.data.DataLoader): Dataloader with training data.
        val_dl (torch.utils.data.DataLoader): Dataloader with validation data.
        hparams (dict): Model hyperparameters.
        trainer (pytorch_lightning.Trainer): Trainer object.
        run_id (optional, str): Unique id to identify the run. If None, will generate an ID containing the current datetime.
        save_model (bool): Whether to save the trained model weights to disk. Defaults to False.

    Returns:
        Classifier: Trained model.
    """
    # generate run_id if None is passed
    if not run_id:
        run_id = generate_run_id()

    wandb.init(
        reinit=True,
        project="slap-gnn",
        name=run_id,
        group="single_point_training",
        config=hparams,
    )

    model = load_model(hparams)

    # run training
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    # optionally, save model weights
    if save_model:
        trainer.save_checkpoint(
            filepath=LOG_DIR / run_id / "model_checkpoints", weights_only=True
        )

    wandb.finish()
    return model


def predict(model, data, trainer):
    """
    Predicts on given data with given model.

    Args:
        model (Classifier): Trained model
        data (torch.utils.data.DataLoader): DataLoader with data for prediction.

    Returns:
        dict: Dictionary of predictions
    """
    probs = torch.cat(trainer.predict(model, dataloaders=data))
    # obtain labels from probabilities at 0.5 threshold
    labels = (probs > 0.5).int()
    return labels


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    config = get_config(args.config)
    # load data
    data = SLAPDataset(
        name=config["data_name"],
        raw_dir=DATA_ROOT,
        reaction=config["reaction"],
        smiles_columns=("SMILES",),
        label_column="targets",
        graph_type=config["graph_type"],
        global_features=config["global_features"],
        featurizers=config["featurizers"],
    )
    # split data into train and validation
    train_size = int(0.9 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])

    # create dataloaders
    train_dl = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_dl = DataLoader(
        val_data,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # update config with data processing specifics
    config["atom_feature_size"] = data.atom_feature_size
    config["bond_feature_size"] = data.bond_feature_size
    config["global_feature_size"] = data.global_feature_size
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        log_every_n_steps=1,
        default_root_dir=PROJECT_DIR,
        accelerator=config["accelerator"],
    )
    model = train(train_dl, val_dl, config, trainer)

    predict_data = SLAPDataset(
        name="validation_plate_featurized_partial.csv",
        raw_dir=DATA_ROOT,
        reaction=config["reaction"],
        smiles_columns=("reactionSMILES",),
        label_column=None,
        graph_type=config["graph_type"],
        global_features=config["global_features"],
        featurizers=config["featurizers"],
    )

    predict_dl = DataLoader(
        predict_data, batch_size=32, collate_fn=collate_fn, shuffle=False
    )

    predictions = predict(model, predict_dl, trainer)
    # save predictions to text file
    with open("predictions.txt", "w") as f:
        for i in predictions.tolist():
            f.write(str(i) + "\n")

    print(predictions)
