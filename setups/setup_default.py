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
from lightning.pytorch.loggers import WandbLogger as Logger
from lightning.pytorch.callbacks import ModelSummary, DeviceStatsMonitor, ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging

# Custom modules
from libraries.Types import Chromosome
from libraries.GeneticAlgorithm import GeneticAlgorithm
from libraries.Operators import Crossover, Mutation, Selection, build_custom_mutation
from libraries.models import ACTIVATIONS, MLP

# Set the seed for repro
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#
#       DATA HANDLING
# 
def load_data(path):
    # Load the data and split it into training and validation sets
    df = pd.read_parquet(path)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=SEED)
    
    # Extract numpy arrays
    X_train, y_train = df_train.drop(columns=['Target']).values, df_train['Target'].values.reshape(-1, 1)
    X_val, y_val = df_val.drop(columns=['Target']).values, df_val['Target'].values.reshape(-1, 1)
    X_test, y_test = df_test.drop(columns=['Target']).values, df_test['Target'].values.reshape(-1, 1)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def build_data_loaders(batch_size, D_train, D_val, D_test):
    # Create PyTorch tensors
    X_train, y_train = torch.tensor(D_train[0], dtype=torch.float32), torch.tensor(D_train[1], dtype=torch.float32)
    X_val, y_val = torch.tensor(D_val[0], dtype=torch.float32), torch.tensor(D_val[1], dtype=torch.float32)
    X_test, y_test = torch.tensor(D_test[0], dtype=torch.float32), torch.tensor(D_test[1], dtype=torch.float32)
    # Create PyTorch data loaders
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


#
#       ALGORITHM COMPONENTS
#
Chromosome.set_domain({})

# Population initialization
def population_initializer(population_size):
    return [Chromosome() for _ in range(population_size)]

# Selection operator
selection_operator = NotImplementedError("You must implement the selection operator")
# Mutation operator
mutation_operator = NotImplementedError("You must implement the mutation operator")
# Crossover pairing
crossover_pairing = Crossover.half_pairs
# Crossover operator
crossover_operator = NotImplementedError("You must implement the crossover operator")
# Termination condition
termination_condition = lambda fitness_values: False



#
#       EVALUATION
# 

# Function that computes the fitness of an individual
def evaluate_individual(
    chromosome,
    train_loader,
    val_loader,
    test_loader,
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
        n_layers    = chromosome['n_layers'],
        activation  = activation,
        dropout     = chromosome['dropout'],
    )
    
    # Define useful callbacks
    model_summary = ModelSummary(max_depth=-1)
    device_monitor = DeviceStatsMonitor()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    model_checkpoint = ModelCheckpoint(
        monitor='val_mse',
        mode='min',
        save_top_k=1,
        dirpath='checkpoints/',
        filename='best-model-{epoch:02d}-{val_mse:.2f}',
        verbose=True,
    )
    early_stopping = EarlyStopping(
        monitor='val_mse',
        mode='min',
        patience=10,
        verbose=True,
    )
    swa = StochasticWeightAveraging(swa_lrs=1e-2)

    # Set up the trainer
    trainer = L.Trainer(
        max_epochs = n_epochs,
        logger = logger,
        # Gradient Clipping
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        # Callbacks
        callbacks = [
            model_summary,
            device_monitor,
            lr_monitor,
            model_checkpoint,
            early_stopping,
            swa
        ],
        # Distributed computing
        accelerator     = 'gpu',
        # num_nodes       = NUM_NODES,
        # strategy        = STRATEGY,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)
    last_val_loss = trainer.callback_metrics['val_mse'].item()

    # Test the model
    trainer.test(ckpt_path="last", dataloaders=test_loader)

    # Finish the logger
    wandb.finish()

    # Return last validation loss
    return -last_val_loss

# Function that evaluates the fitness of each individual in the population
def evaluate_population(
    chromosomes,
    data,
    n_epochs  = 10,
    loggers   = None,
):
    if loggers is None:
        loggers = [None] * len(chromosomes)
    train_loader, val_loader, test_loader = data
    return [evaluate_individual(chromosome, train_loader, val_loader, test_loader, n_epochs, logger) for logger, chromosome in zip(loggers, chromosomes)]


