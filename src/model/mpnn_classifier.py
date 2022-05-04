import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm

from src.layer.mpnn import MPNNEncoder
from src.layer.ffn import FFN


class DMPNNModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        params = {}
        self.encoder = self.init_encoder(params)
        self.decoder = self.init_decoder(params)

    def init_encoder(self, params):
        return MPNNEncoder(atom_feature_size=self.hparams["atom_feature_size"],
                           bond_feature_size=self.hparams["bond_feature_size"],
                           **self.hparams.encoder
                           )

    def init_decoder(self, params):
        return FFN(in_size=self.hparams.encoder["hidden_size"],
                   out_size=1,
                   **self.hparams.decoder,
                   )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def _get_preds_loss_metrics(self, batch):

        # predict for batch
        cgr_batch, y = batch
        embedding = self.encoder(cgr_batch)
        y_hat = self.decoder(embedding)

        # calculate loss
        loss = self.calc_loss(y_hat, y)

        # calculate additional metrics
        acc = tm.Accuracy()
        auc = tm.AUROC()
        prec = tm.Precision()
        rec = tm.Recall()
        return y_hat, loss, {"accuracy": acc(y_hat, y), "AUROC": auc(y_hat, y), "precision": prec(y_hat, y), "recall": rec(y_hat, y)}

    def training_step(self, batch, batch_idx):
        preds, loss, metrics = self._get_preds_loss_metrics(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for k, v in metrics.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, metrics = self._get_preds_loss_metrics(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for k, v in metrics.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        preds, loss, metrics = self._get_preds_loss_metrics(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.optimizer["lr"],
            weight_decay=self.hparams.optimizer["weight_decay"],
        )

        # learning rate scheduler
        scheduler = self._config_lr_scheduler(optimizer)

        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/loss",
            }

    def _config_lr_scheduler(self, optimizer):

        scheduler_name = self.hparams.optimizer["lr_scheduler"]["scheduler_name"].lower()

        if scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.4, patience=50, verbose=True
            )
        elif scheduler_name == "exp_with_linear_warmup":
            lr_min = self.hparams.optimizer["lr_scheduler"]["lr_min"]
            lr_max = self.hparams.optimizer["lr"]
            start_factor = lr_min / lr_max
            end_factor = 1
            n_warmup_epochs = self.hparams.optimizer["lr_scheduler"]["lr_warmup_step"]

            gamma = start_factor ** (1 / (self.hparams.optimizer["lr_scheduler"]["epochs"] - n_warmup_epochs))

            def lr_foo(epoch):
                if epoch <= self.hparams.optimizer["lr_scheduler"]["lr_warmup_step"]:
                    # warm up lr
                    lr_scale = start_factor + (epoch * (end_factor - start_factor / end_factor) / n_warmup_epochs)

                else:
                    lr_scale = gamma ** (epoch - n_warmup_epochs)

                return lr_scale

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_foo
            )

        elif scheduler_name == "none":
            scheduler = None
        else:
            raise ValueError(f"Not supported lr scheduler: {self.hparams.optimizer['lr_scheduler']}")

        return scheduler

    def calc_loss(self, preds, truth):
        # TODO here I might want a different loss or expose the choice through hyperparameters

        loss = F.binary_cross_entropy_with_logits(
            preds,
            truth.to(torch.float),  # input label is int for metric purpose
            reduction="mean",
        )
        return loss
