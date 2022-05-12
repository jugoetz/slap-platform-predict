import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.model.mpnn_classifier import DMPNNModel
from src.util.definitions import PROJECT_DIR, LOG_DIR
from src.util.logging import generate_run_id
from src.data.dataloader import collate_fn


def run_training(hparams, data, trainer, save_model=False):
    """Convenience wrapper around train() to train a model on a single train-test split"""

    # split data
    data_train, data_val = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)

    # instantiate DataLoaders
    train_dl = DataLoader(data_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(data_val, batch_size=32, collate_fn=collate_fn)

    # run training
    metrics, trained_model = train(train_dl, val_dl, hparams, trainer, save_model=save_model)

    return metrics, trained_model


def train(train, val, hparams, trainer, run_id=None, test=None, return_test_metrics=False, save_model=False):
    """
    Trains a model on a given data split with one set of hyperparameters. By default, returns the evaluation metrics
    on the validation set.

    Args:
        train (torch.utils.data.DataLoader): Training data
        val: (torch.utils.data.DataLoader): Validation data
        test: (torch.utils.data.DataLoader, optional): Test data. Only used if return_test_metrics is True.
        hparams (dict): Model hyperparameters
        run_id (optional, str): Unique id to identify the run. If None, will generate an ID containing the current datetime.
        return_test_metrics (bool): Whether to return metrics on test set. If True, test has to be given.
            Defaults to False.
        save_model (bool): Whether to save the trained model weights to disk. Defaults to False.

    Returns:
        dict: Dictionary of validation metrics and, if return_test_metrics is True, additionally test metrics
        DMPNNModel: Trained model
    """
    # generate run_id if None is passed
    if not run_id:
        run_id = generate_run_id()

    # check if a test set has been passed
    if return_test_metrics:
        if test is None:
            raise ValueError("When return_test_metrics is True, as test dataloader has to be passed.")

    # initialize model
    model = DMPNNModel(**hparams)

    # run training
    trainer.fit(model, train_dataloaders=train, val_dataloaders=val)

    # optionally, save model weights
    if save_model:
        trainer.save_checkpoint(filepath=LOG_DIR / run_id / "model_checkpoints", weights_only=True)

    # optionally, run test set
    if return_test_metrics:
        trainer.test(model, test)

    return_metrics = {k: v for k, v in trainer.logged_metrics.items() if k.startswith('val')}
    if return_test_metrics:
        test_metrics = {k: v for k, v in trainer.logged_metrics.items() if k.startswith('test')}
        return_metrics.update(test_metrics)
    return return_metrics, model




