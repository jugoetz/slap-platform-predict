from collections import defaultdict

import torch
from sklearn.model_selection import KFold, ShuffleSplit
from torch.utils.data import DataLoader

from src.data.dataloader import collate_fn
from src.train import train
from src.util.logging import generate_run_id


def cross_validate(hparams, data, n_folds=5, strategy="KFold", save_models=False):
    """
    Trains a model under cross-validation. Returns the validation metrics' mean and std.

    Args:
        hparams (dict): Model configuration. See model for details.
        data (torch.utils.data.Dataset): Data set to run CV on.
        n_folds (int): Number of folds. If 1, a 4:1 ShuffleSplit will be used regardless of strategy.
        strategy (str): CV strategy. Supported: {"KFold"}. Defaults to "KFold".

    Returns:
        dict: Metrics returned by train(), aggregated across folds (mean and standard deviation).
    """
    # set up splitter
    if n_folds == 1:
        splitter = ShuffleSplit(test_size=0.2, random_state=42)
    elif strategy == "KFold":
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        raise ValueError(f"Invalid strategy {strategy}")

    # generate run_id
    cv_run_id = generate_run_id()

    # iterate folds
    metrics = defaultdict(list)
    for i, (train_idx, val_idx) in enumerate(splitter.split(list(range(len(data))))):
        if i >= n_folds:
            break
        data_train = [data[i] for i in train_idx]
        data_val = [data[i] for i in val_idx]
        train_dl = DataLoader(data_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_dl = DataLoader(data_val, batch_size=32, collate_fn=collate_fn)
        fold_metrics = train(train_dl, val_dl, hparams, run_id=f"{cv_run_id}_fold{i}", save_model=save_models)
        for k, v in fold_metrics.items():
            metrics[k].append(v)

    # aggregate fold metrics
    metrics = {k: torch.stack(v) for k, v in metrics.items()}
    metrics_return = {k + "_mean": torch.mean(v) for k, v in metrics.items()}
    metrics_std = {k + "_std": torch.std(v) for k, v in metrics.items()}
    metrics_return.update(metrics_std)
    return metrics_return
