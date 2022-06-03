from collections import defaultdict

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.data.dataloader import collate_fn
from src.train import train
from src.util.definitions import PROJECT_DIR
from src.util.io import index_from_file
from src.util.logging import generate_run_id


def cross_validate(hparams, data, trainer, n_folds=5, strategy="KFold", save_models=False, return_fold_metrics=False):
    """
    Trains a model under cross-validation. Returns the validation metrics' mean and std.

    Args:
        hparams (dict): Model configuration. See model for details.
        data (torch.utils.data.Dataset): Data set to run CV on.
        n_folds (int): Number of folds. If 1, a 4:1 ShuffleSplit will be used regardless of strategy.
        strategy (str): CV strategy. Supported: {"KFold"}. Defaults to "KFold".
        save_models (bool, optional): Whether to save a model checkpoint after training
        return_fold_metrics (bool, optional): Whether to additionally return full train and val metrics for all folds

    Returns:
        dict: Validation metrics, aggregated across folds (mean and standard deviation).
        dict: All metrics returned by train(), for all folds. Only returned if return_fold_metrics is True.
    """
    # set up splitter
    if n_folds < 2:
        raise ValueError("n_folds must be > 1 for cross-validation.")

    if strategy == "KFold":
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        raise ValueError(f"Invalid strategy '{strategy}'")

    # generate run_id
    cv_run_id = generate_run_id()

    # iterate folds
    metrics = defaultdict(list)
    for i, (train_idx, val_idx) in enumerate(splitter.split(list(range(len(data))))):
        if i >= n_folds:
            break  # exit loop for "endless" splitters like ShuffleSplit
        data_train = [data[i] for i in train_idx]
        data_val = [data[i] for i in val_idx]
        train_dl = DataLoader(data_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_dl = DataLoader(data_val, batch_size=32, collate_fn=collate_fn)
        fold_metrics, _ = train(train_dl, val_dl, hparams, trainer, run_id=f"{cv_run_id}_fold{i}",
                                save_model=save_models)
        for k, v in fold_metrics.items():
            metrics[k].append(v)

    # aggregate fold metrics
    metrics = {k: torch.stack(v) for k, v in metrics.items()}
    metrics_return = {k + "_mean": torch.mean(v) for k, v in metrics.items()}
    metrics_std = {k + "_std": torch.std(v) for k, v in metrics.items()}
    metrics_return.update(metrics_std)

    if return_fold_metrics:
        return metrics_return, metrics
    else:
        return metrics_return


def cross_validate_predefined(hparams, data, split_files, save_models=False, return_fold_metrics=False):
    """
    Trains a model under cross-validation. Returns the validation metrics' mean/std and if test sets are given,
    test mean/std.

    The CV splits must be supplied as files containing indices (see arg split_files).

    Function will report evaluation metrics on all sets, i.e. val, and any sets starting with "test".
    Only val may be used to compare models before selection. To avoid computing test scores,
    do not pass test set indices.

    Args:
        hparams (dict): Model configuration. See model for details.
        data (torch.utils.data.Dataset): Data set to run CV on.
        split_files (list): Splits to use. Expects a list of dictionaries, where each dictionary represents one fold
            e.g. [{"train": <path_to_train>, "val": <path_to_val>, "test": <path_to_test>}, ...]
            the dictionaries must contain keys "train" and "val" and can contain an arbitrary number of keys starting
            with "test"
        save_models (bool, optional): Whether to save a model checkpoint after training
        return_fold_metrics (bool, optional): Whether to additionally return full train and val metrics for all folds

    Returns:
        dict: Validation metrics, aggregated across folds (mean and standard deviation).
        dict: All metrics returned by train(), for all folds. Only returned if return_fold_metrics is True.
    """

    # generate run_id
    cv_run_id = generate_run_id()

    # dict to hold fold metrics
    metrics = defaultdict(list)

    # iterate folds
    for i, fold in enumerate(split_files):

        # each fold needs a new trainer instance b/c it can only fit once
        fold_run_id = f"{cv_run_id}_fold{i}"
        trainer_fold = pl.Trainer(max_epochs=hparams["training"]["max_epochs"], log_every_n_steps=1,
                                  default_root_dir=PROJECT_DIR, logger=WandbLogger(name=fold_run_id, group=cv_run_id),
                                  accelerator=hparams["accelerator"])

        # load indices from file
        idx = {k: index_from_file(v) for k, v in fold.items()}

        # instantiate DataLoaders
        data_splitted = {k: [data[i] for i in v] for k, v in idx.items()}
        train_dl = DataLoader(data_splitted["train"], batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_dl = DataLoader(data_splitted["val"], batch_size=32, collate_fn=collate_fn)
        test_dls = {k: DataLoader(v, batch_size=32, collate_fn=collate_fn) for k, v in data_splitted.items() if
                    k.startswith("test")}

        # train and validate model
        fold_metrics_val, model = train(train_dl, val_dl, hparams, trainer_fold, run_id=fold_run_id,
                                        save_model=save_models)
        for k, v in fold_metrics_val.items():
            metrics[k].append(v)

        # test model (if any test sets were given)
        for test_name, test_dl in test_dls.items():
            trainer_fold.test(model, test_dl, ckpt_path="best")
            for k, v in trainer_fold.logged_metrics.items():
                if k.startswith('test'):
                    metrics[k.replace("test", test_name)].append(v)
        wandb.finish()

    # aggregate fold metrics
    metrics = {k: torch.stack(v) for k, v in metrics.items()}
    metrics_return = {k + "_mean": torch.mean(v) for k, v in metrics.items()}
    metrics_std = {k + "_std": torch.std(v) for k, v in metrics.items()}
    metrics_return.update(metrics_std)

    if return_fold_metrics:
        return metrics_return, metrics
    else:
        return metrics_return
