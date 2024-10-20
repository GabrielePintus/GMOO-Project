import random
import numpy as np
import pandas as pd

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

# Logging
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelSummary, DeviceStatsMonitor, ModelCheckpoint

# Custom modules
from libraries.Types import Chromosome
from libraries.Operators import Crossover, Mutation, Selection, build_custom_mutation
from libraries.models import ACTIVATIONS, MLP
from libraries.GeneticAlgorithm import GeneticAlgorithm

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Parallelism and Distributed Computing
# from mpi4py import MPI
# from threading import Thread, Semaphore
# NUM_NODES = 1
# NUM_GPUS = 1

# Preliminary setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)




#
#       DATA HANDLING
# 
def load_data(path):
    # Load the data and split it into training and validation sets
    df = pd.read_parquet(path)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
    
    # Extract numpy arrays
    X_train, y_train = df_train.drop(columns=['Target']).values, df_train['Target'].values.reshape(-1, 1)
    X_test, y_test = df_test.drop(columns=['Target']).values, df_test['Target'].values.reshape(-1, 1)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test
    
    


def build_data_loaders(batch_size, X_train, y_train, X_val, y_val):
    # Create PyTorch data loaders
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


#
#       ALGORITHM COMPONENTS
#
Chromosome.set_domain({
    'hidden_size': (4, 32),
    'activation': [0, 1, 2, 3, 4],
    'dropout': (0, 0.5),
})

# Population initialization
def population_initializer(population_size):
    return [Chromosome() for _ in range(population_size)]

# Mutation operator
# custom_mutation = build_custom_mutation({
#     slice(0, 1) : lambda x: Mutation.uniform_mutation(x, 1, -4, 4),
#     slice(1, 2) : lambda x: Mutation.uniform_mutation(x, 1, -2, 2),
#     slice(2, 3) : lambda x: Mutation.gaussian_mutation(x, 1, 0, 0.15),
# })
custom_mutation = build_custom_mutation({
    (0, 1) : lambda x: Mutation.uniform_mutation(x, 1, -4, 4),
    (1, 2) : lambda x: Mutation.uniform_mutation(x, 1, -2, 2),
    (2, 3) : lambda x: Mutation.gaussian_mutation(x, 1, 0, 0.15),
})


#
#       Training
#
def fitness_evaluator(
    chromosomes,
    data,
    n_epochs  = 10,
    loggers   = None
):
    if loggers is None:
        loggers = [None] * len(chromosomes)
    train_loader, val_loader = data
    return [evaluate_fitness(chromosome, train_loader, val_loader, n_epochs, logger) for logger, chromosome in zip(loggers, chromosomes)]

def evaluate_fitness(
    chromosome,
    train_loader,
    val_loader,
    n_epochs  = 10,
    logger    = None
):
    chromosome = chromosome.to_dict()

    # Instantiate the model
    activation = ACTIVATIONS[chromosome['activation']]
    model = MLP(
        input_size  = train_loader.dataset.tensors[0].shape[1],
        hidden_size = chromosome['hidden_size'],
        output_size = 1,
        activation  = activation,
        dropout     = chromosome['dropout'],
    )


    # Log Hyperparameters
    
    # Define useful callbacks
    model_summary = ModelSummary(max_depth=-1)
    device_monitor = DeviceStatsMonitor()

    # Set up the trainer
    trainer = L.Trainer(
        max_epochs = n_epochs,
        logger = logger,
        callbacks = [
            model_summary,
            device_monitor,
        ],
        accelerator='gpu',
        # devices=NUM_GPUS,
        # num_nodes=NUM_NODES,
        # strategy='ddp',
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Finish the logger
    wandb.finish()

    
    # Return last validation loss
    return -model.val_metrics['mse'].compute()

