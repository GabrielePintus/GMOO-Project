import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils.Types import Chromosome
from utils.Operators import Crossover, Mutation, Selection, build_custom_mutation
import matplotlib.pyplot as plt
from utils.models import MLP
import torch
from utils.GeneticAlgorithm import GeneticAlgorithm
import torch.nn as nn
from mpi4py import MPI
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Preliminary setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
#       DATA HANDLING
# 
def load_data(path):
    # Load the data and split it into training and validation sets
    df = pd.read_parquet(path)
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=SEED)

    # Split the data into features and target
    X_train, y_train = df_train.drop(columns=['Target']), df_train['Target'].values.reshape(-1, 1)
    X_val, y_val = df_val.drop(columns=['Target']), df_val['Target'].values.reshape(-1, 1)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val


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

custom_mutation = build_custom_mutation({
    slice(0, 1) : lambda x: Mutation.uniform_mutation(x, 1, -4, 4),
    slice(1, 2) : lambda x: Mutation.uniform_mutation(x, 1, -2, 2),
    slice(2, 3) : lambda x: Mutation.gaussian_mutation(x, 1, 0, 0.15),
})

def population_initializer(population_size):
    return [Chromosome() for _ in range(population_size)]

def evaluate_fitness(chromosome, train_loader, val_loader, n_epochs=10, clip_grad=1.0):
    chromosome = chromosome.to_dict()
    model = MLP(input_size=5, output_size=1, **chromosome).to(device)
    
    initial_lr = 1e-2
    final_lr = 1e-4
    gamma = (final_lr / initial_lr) ** (1 / n_epochs)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    criterion = nn.MSELoss()
    
    for _ in range(n_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                val_loss += criterion(y_pred, y)
        val_loss /= len(val_loader)
    
    return -val_loss.item()

def fitness_evaluator(chromosomes, data):
    train_loader, val_loader = data
    return [evaluate_fitness(chromosome, train_loader, val_loader) for chromosome in chromosomes]


if __name__ == '__main__':
    ga = GeneticAlgorithm(
        population_initializer  = population_initializer,
        fitness_evaluator       = fitness_evaluator,
        selection_operator      = lambda pop, fitness : Selection.tournament_selection(pop , fitness, 2),
        crossover_pairing       = Crossover.half_pairs,
        crossover_operator      = Crossover.uniform_crossover,
        mutation_operator       = custom_mutation,
        termination_condition   = lambda fitness_values : np.max(fitness_values) == 0
    )

    X_train, y_train, X_val, y_val = load_data('datasets/sinewave.parquet')
    train_loader, val_loader = build_data_loaders(128, X_train, y_train, X_val, y_val)

    # Run the algorithm
    ga.run(
        data = (train_loader, val_loader),
        population_size= 4,
        max_generations=20,
    )