import pickle
from copy import deepcopy

import numpy as np
import wandb
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    log_loss,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.model.classifier import DMPNNModel, GCNModel, FFNModel
from src.util.definitions import LOG_DIR
from src.util.logging import generate_run_id, concatenate_to_dict_keys
from src.data.dataloader import collate_fn


def run_training(hparams, data, trainer, save_model=False):
    """Convenience wrapper around train() to train a model on a single train-test split"""

    # split data
    data_train, data_val = train_test_split(
        data, test_size=0.2, shuffle=True, random_state=42
    )

    # instantiate DataLoaders
    train_dl = DataLoader(
        data_train, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_dl = DataLoader(data_val, batch_size=32, collate_fn=collate_fn)

    # run training
    metrics, trained_model = train(
        train_dl, val_dl, hparams, trainer, save_model=save_model
    )

    return metrics, trained_model


def train(
    train,
    val,
    hparams,
    trainer,
    run_id=None,
    group_run_id=None,
    test=None,
    save_model=False,
):
    """
    Trains a model on a given data split with one set of hyperparameters. By default, returns the evaluation metrics
    on the validation set.

    Args:
        train (torch.utils.data.DataLoader): Training data
        val: (torch.utils.data.DataLoader): Validation data
        hparams (dict): Model hyperparameters
        run_id (optional, str): Unique id to identify the run. If None, will generate an ID containing the current datetime.
        group_run_id (optional, str): Id to identify the run group. Default None.
        test: (dict, optional): Test data. Dictionary of dataloaders. If given, function will return test score(s).
        save_model (bool): Whether to save the trained model weights to disk. Defaults to False.

    Returns:
        dict: Dictionary of validation metrics and, if return_test_metrics is True, additionally test metrics
        DMPNNModel: Trained model
    """
    # generate run_id if None is passed
    if not run_id:
        run_id = generate_run_id()

    wandb.init(
        reinit=True, project="slap-gnn", name=run_id, group=group_run_id, config=hparams
    )

    # initialize model
    if hparams["encoder"]["type"] == "D-MPNN":
        model = DMPNNModel(**hparams)
    elif hparams["encoder"]["type"] == "GCN":
        model = GCNModel(**hparams)
    else:
        model = FFNModel(**hparams)

    # run training
    trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
    metrics = {k: v for k, v in trainer.logged_metrics.items()}
    # optionally, save model weights
    if save_model:
        trainer.save_checkpoint(
            filepath=LOG_DIR / run_id / "model_checkpoints", weights_only=True
        )

    # optionally, run test set
    if test:
        for test_name, test_dl in test.items():
            trainer.test(model, test_dl, ckpt_path="best")
            for k, v in trainer.logged_metrics.items():
                if k.startswith("test"):
                    metrics[k.replace("test", test_name)] = v

    wandb.log(metrics)
    wandb.finish()
    return metrics, model


def train_sklearn(
    train, val, hparams, run_id=None, group_run_id=None, test=None, save_model=False
):
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

    wandb.init(
        reinit=True, project="slap-gnn", name=run_id, group=group_run_id, config=hparams
    )

    # initialize model
    if hparams["decoder"]["type"] == "LogisticRegression":
        model = LogisticRegression(**hparams["decoder"]["LogisticRegression"])
    elif hparams["decoder"]["type"] == "XGB":
        model = XGBClassifier(**hparams["decoder"]["XGB"])
    else:
        raise ValueError("Invalid model type")

    # get training and validation data
    train_graphs, train_global_features, train_labels = map(list, zip(*train))
    val_graphs, val_global_features, val_labels = map(list, zip(*val))

    if hparams["encoder"]["type"] == "global_features":
        X_train = train_global_features
        X_val = val_global_features
    else:
        raise ValueError("Invalid encoder type for sklearn model")

    # run training
    model.fit(X_train, train_labels)

    # evaluate on training set
    train_pred = model.predict_proba(X_train)
    train_metrics = concatenate_to_dict_keys(
        calculate_metrics(train_labels, train_pred, pred_proba=True), prefix="train/"
    )

    # evaluate on validation set
    val_pred = model.predict_proba(X_val)
    val_metrics = concatenate_to_dict_keys(
        calculate_metrics(val_labels, val_pred, pred_proba=True), prefix="val/"
    )

    # optionally, save model
    if save_model:
        with open(LOG_DIR / run_id / "model_checkpoints" / "model.pkl", "wb") as f:
            pickle.dump(model, f)

    # optionally, run test set
    if test:
        test_metrics = {}
        for k, v in test.items():
            test_graphs, test_global_features, test_labels = map(list, zip(*v))
            if hparams["encoder"]["type"] == "global_features":
                X_test = test_global_features
            else:
                raise ValueError("Invalid encoder type for sklearn model")
            test_pred = model.predict_proba(X_test)
            test_metrics.update(
                concatenate_to_dict_keys(
                    calculate_metrics(test_labels, test_pred, pred_proba=True), f"{k}/"
                )
            )

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

    if pred_proba is True:
        y_prob = deepcopy(y_pred)
        y_pred = np.argmax(y_pred, axis=1)

    metrics = {
        k: v
        for k, v in zip(
            ["precision", "recall", "f1"],
            precision_recall_fscore_support(
                y_true, y_pred, average="binary", pos_label=1
            ),
        )
    }
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["AUROC"] = roc_auc_score(y_true, y_pred)
    metrics["loss"] = log_loss(y_true, y_prob) if pred_proba else None

    if detailed is True:
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        metrics["roc_curve"] = roc_curve(y_true, y_pred)
        metrics["pr_curve"] = precision_recall_curve(y_true, y_pred)
    return metrics
