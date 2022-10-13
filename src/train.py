import pickle

import wandb
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.evaluate import calculate_metrics
from src.model.classifier import load_model
from src.util.definitions import LOG_DIR, CKPT_DIR
from src.util.logging import generate_run_id, concatenate_to_dict_keys


def train(
    train_dl,
    val_dl,
    hparams,
    test_dls=None,
    run_id=None,
    run_group=None,
    return_fold_metrics=False,
):
    """
    Trains a model on given data with one set of hyperparameters. Training and validation metrics (as specified in
    the model class) are logged to wandb.

    Args:
        train_dl (torch.utils.data.DataLoader): Dataloader with training data.
        val_dl (torch.utils.data.DataLoader): Dataloader with validation data.
        hparams (dict): Model hyperparameters.
        test_dls (optional, dict): Dictionary of dataloaders with test data. If given, test metrics will be returned.
        run_id (optional, str): Unique id to identify the run. If None, will generate an ID containing the current datetime.
        run_group (optional, str): Name to identify the run group. Default None.
        return_fold_metrics (bool, optional): Whether to return train and val metrics. Defaults to False.

    Returns:
        str: run_id that identifies the run/model
        dict: Dictionary of training and validation metrics. Only returned if return_fold_metrics is True.
    """
    # generate run_id if None is passed
    if not run_id:
        run_id = generate_run_id()

    # set run group name
    if not run_group:
        run_group = "single_run"

    # set up trainer
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss", mode="min", dirpath=CKPT_DIR / run_id, filename="best"
        )
    ]

    trainer = pl.Trainer(
        max_epochs=hparams["training"]["max_epochs"],
        log_every_n_steps=1,
        default_root_dir=LOG_DIR / "checkpoints",
        accelerator=hparams["accelerator"],
        callbacks=callbacks,
    )

    wandb.init(
        reinit=True,
        project="slap-gnn",
        name=run_id,
        group=run_group,
        config=hparams,
    )

    model = load_model(hparams)

    # run training
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # dict for logged metrics
    metrics = {k: v for k, v in trainer.logged_metrics.items()}

    # optionally, run test
    if test_dls:
        for test_name, test_dl in test_dls.items():
            trainer.test(model, test_dl, ckpt_path="best")
            for k, v in trainer.logged_metrics.items():
                if k.startswith("test"):
                    metrics[k.replace("test", test_name)] = v

    wandb.log(metrics)
    wandb.finish()

    if return_fold_metrics:
        return run_id, metrics
    else:
        return run_id


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
