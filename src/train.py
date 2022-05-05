import pytorch_lightning as pl

from src.model.mpnn_classifier import DMPNNModel
from src.util.definitions import PROJECT_DIR, LOG_DIR
from src.util.logging import generate_run_id


def train(train, val, hparams, run_id=None, test=None, return_test_metrics=False, save_model=False):
    """
    Args:
        train (torch.utils.data.DataLoader): Training data
        val: (torch.utils.data.DataLoader): Validation data
        test: (torch.utils.data.DataLoader, optional): Test data. Only used if return_test_metrics is True.
        hparams (dict): Model hyperparameters
        run_id (optional, str): Unique id to identify the run. If None, will generate an ID containing the current datetime and a random integer.
        return_test_metrics (bool): Whether to return metrics on test set. If True, test has to be given.
            Defaults to False.
        save_model (bool): Whether to save the trained model weights to disk. Defaults to False.

    Returns:
        dict: Dictionary of validation metrics and, if return_test_metrics is True, additionally test metrics

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
    # initialize trainer
    trainer = pl.Trainer(max_epochs=hparams["training"]["max_epochs"], log_every_n_steps=1,
                         default_root_dir=PROJECT_DIR, logger=False)
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
    return return_metrics




