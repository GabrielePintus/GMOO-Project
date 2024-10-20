from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn
import torchmetrics
import lightning as L
from libraries.Types import Chromosome
from torch.optim.lr_scheduler import StepLR
from torch.functional import F
import numpy as np


ACTIVATIONS = (nn.ReLU(), nn.LeakyReLU(), nn.ELU(), nn.Tanh(), nn.Sigmoid())


class RandomForest(RandomForestClassifier):
    def __init__(self, **kwargs):
        # The last position is the criterion
        # Map 0 to 'gini' and 1 to 'entropy'
        kwargs['criterion'] = ['gini', 'entropy'][kwargs['criterion']]
        super().__init__(**kwargs)
        

class MLP(L.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size, 
        output_size,
        activation,
        dropout
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        # Metrics
        self.metrics = {
            'mse': torchmetrics.MeanSquaredError(),
            'mae': torchmetrics.MeanAbsoluteError(),
            'R2': torchmetrics.R2Score(),
        }
        self.train_metrics = torchmetrics.MetricCollection(self.metrics, prefix='train_')
        self.val_metrics = torchmetrics.MetricCollection(self.metrics, prefix='val_')
        # Save the hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)

        # Log the metrics
        metric_values = self.train_metrics(y_hat, y)
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
        self.log('test_loss', loss)
        return loss

