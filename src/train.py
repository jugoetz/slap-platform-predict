import pickle

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve, \
    precision_recall_curve, auc, classification_report, log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier

from src.model.classifier import DMPNNModel, GCNModel
from src.util.definitions import LOG_DIR
from src.util.logging import generate_run_id, concatenate_to_dict_keys
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
        hparams (dict): Model hyperparameters
        run_id (optional, str): Unique id to identify the run. If None, will generate an ID containing the current datetime.
        test: (torch.utils.data.DataLoader, optional): Test data. Only used if return_test_metrics is True.
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
    if hparams["encoder"]["type"] == "D-MPNN":
        model = DMPNNModel(**hparams)
    elif hparams["encoder"]["type"] == "GCN":
        model = GCNModel(**hparams)
    else:
        raise ValueError("Invalid model type")

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


def train_sklearn(train, val, hparams, run_id=None, group_run_id=None, test=None, save_model=False):
    """
    Trains a sklearn model on a given data split with one set of hyperparameters. By default, returns the evaluation
    metrics on the validation set.

    Args:
        train (torch.utils.data.DataLoader): Training data
        val: (torch.utils.data.DataLoader): Validation data
        test: (Union[DataLoader, Dict[torch.utils.data.DataLoader]], optional): Test data. If data is given, test metrics will be returned.
        hparams (dict): Model hyperparameters
        run_id (optional, str): Unique id to identify the run. If None, will generate an ID containing the current datetime.
            Defaults to None.
        group_run_id (optional, str): Id to identify the run group. Default None.
        save_model (bool): Whether to save the trained model weights to disk. Defaults to False.

    Returns:
        dict: Dictionary of validation metrics and, test DataLoader(s) are passed, additionally test metrics
        Model: Trained model
    """
    # generate run_id if None is passed
    if not run_id:
        run_id = generate_run_id()

    wandb.init(reinit=True, project="slap-gnn", name=run_id, group=group_run_id, config=hparams)

    # initialize model
    if hparams["decoder"]["type"] == "Ridge":
        model = RidgeClassifier(**hparams["decoder"]["Ridge"])
    elif hparams["decoder"]["type"] == "XGB":
        model = XGBClassifier(**hparams["decoder"]["XGB"])
    else:
        raise ValueError("Invalid model type")

    # get training and validation data
    train_graphs, train_global_features, train_fingerprints, train_labels = map(list, zip(*train))
    val_graphs, val_global_features, val_fingerprints, val_labels = map(list, zip(*val))

    if hparams["encoder"]["type"] == "RDKit":
        X_train = train_global_features
        X_val = val_global_features
    elif hparams["encoder"]["type"] == "FP":
        X_train = train_fingerprints
        X_val = val_fingerprints
    else:
        raise ValueError("Invalid encoder type for sklearn model")

    # run training
    model.fit(X_train, train_labels)

    # evaluate on training set
    train_pred = model.predict(X_train)
    train_metrics = concatenate_to_dict_keys(calculate_metrics(train_labels, train_pred), prefix="train/")

    # evaluate on validation set
    val_pred = model.predict(X_val)
    val_metrics = concatenate_to_dict_keys(calculate_metrics(val_labels, val_pred), prefix="val/")

    # optionally, save model
    if save_model:
        with open(LOG_DIR / run_id / "model_checkpoints" / "model.pkl", "wb") as f:
            pickle.dump(model, f)

    # optionally, run test set
    if test:
        test_metrics = {}
        for k, v in test.items():
            test_graphs, test_global_features, test_fingerprints, test_labels = map(list, zip(*v))
            if hparams["encoder"]["type"] == "RDKit":
                X_test = test_global_features
            elif hparams["encoder"]["type"] == "FP":
                X_test = test_fingerprints
            else:
                raise ValueError("Invalid encoder type for sklearn model")
            test_pred = model.predict(X_test)
            test_metrics.update(concatenate_to_dict_keys(calculate_metrics(test_labels, test_pred), f"{k}/"))

    return_metrics = {}
    return_metrics.update(train_metrics)
    return_metrics.update(val_metrics)

    if test:
        return_metrics.update(test_metrics)

    wandb.log(return_metrics)
    wandb.finish()
    return return_metrics, model


def calculate_metrics(y_true, y_pred, pred_proba=False, detailed=False):
    """
    Calculate a bunch of metrics for classification problems.
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        pred_proba: Whether y_pred is a probability, or a label. If False, y_pred is assumed to be a label and
            log_loss is not calculated. Defaults to False.
        detailed: Whether to include ROC curve, PR curve, and confusion matrix. Defaults to False.

    Returns:
        dict: Dictionary of metrics

    """

    metrics = {k: v for k, v in
               zip(["precision", "recall", "f1"], precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1))}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["AUROC"] = roc_auc_score(y_true, y_pred)
    if pred_proba is True:
        metrics["loss"] = log_loss(y_true, y_pred)
    if detailed is True:
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        metrics["roc_curve"] = roc_curve(y_true, y_pred)
        metrics["pr_curve"] = precision_recall_curve(y_true, y_pred)
    return metrics
