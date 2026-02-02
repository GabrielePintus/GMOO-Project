from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn
import torchmetrics
import lightning as L
from libraries.Types import Chromosome
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.functional import F
import numpy as np


ACTIVATIONS = [nn.ReLU(), nn.LeakyReLU(), nn.ELU(), nn.GELU(), nn.Tanh(), nn.Sigmoid()]

        

class MLP(L.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size, 
        output_size,
        n_layers,
        activation,
        dropout: float = 0.5,
    ):
        super().__init__()

        use_dropout = dropout > 0.0
        dropout_p = dropout
        layers = [
            nn.Linear(input_size, hidden_size),
            activation,
        ]

        if use_dropout:
            layers.append(nn.Dropout(dropout_p))

        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                activation,
            ])
            if use_dropout:
                layers.append(nn.Dropout(dropout_p))

        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

        # Metrics
        self.metrics = {
            'mse': torchmetrics.MeanSquaredError(),
            'mae': torchmetrics.MeanAbsoluteError(),
            'R2': torchmetrics.R2Score(),
        }
        self.train_metrics = torchmetrics.MetricCollection(self.metrics, prefix='train/')
        self.val_metrics = torchmetrics.MetricCollection(self.metrics, prefix='val/')
        self.test_metrics = torchmetrics.MetricCollection(self.metrics, prefix='test/')

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        initial_lr = 1e-2
        final_lr = 1e-5
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=initial_lr)
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=final_lr,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_mse'
        }

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_pred = self.model(x)
        loss = F.mse_loss(y_pred, y)

        # Log the metrics
        metric_values = self.train_metrics(y_pred, y)
        self.log_dict(metric_values, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.mse_loss(y_pred, y)
        
        # Log the metrics
        metric_values = self.val_metrics(y_pred, y)
        self.log_dict(metric_values, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.mse_loss(y_pred, y)

        # Log the metrics
        metric_values = self.test_metrics(y_pred, y)
        self.log_dict(metric_values, on_epoch=True)

        return loss

