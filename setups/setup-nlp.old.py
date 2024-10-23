# Preliminary setup
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Packages
import torch
from lightning.pytorch.loggers import WandbLogger as Logger
from libraries.GeneticAlgorithm import GeneticAlgorithm



#
#       DATA HANDLING
#
from setup import load_data, build_data_loaders

#
#       ALGORITHM COMPONENTS
#
# Consider we have to tune the following hyperparameters:
# - Number of hidden layers                 Positive integer
# - Number of neurons per hidden layer      Positive integer
# - Activation function                     [ReLU, LeakyReLU, ELU, Tanh, Sigmoid]
# - Dropout rate                            Real number in [0, 0.5]
from libraries.Types import Chromosome
from libraries.models import ACTIVATIONS
Chromosome.set_domain({
    'n_layers': (1, 5),
    'hidden_size': (16, 128),
    'activation': list(range(len(ACTIVATIONS))),
    'dropout': (0, 0.5)
})

# Define the initialization function
from setup import population_initializer

# Define the selection operator
from libraries.Operators import Selection
selection_operator = lambda population, fitness_values: Selection.tournament_selection(population, fitness_values, 3)

# Define crossover pairing
from libraries.Operators import Crossover
crossover_pairing = Crossover.half_pairs

# Define the crossover operator
crossover_operator = Crossover.two_point_crossover

# Define the mutation operator
from libraries.Operators import Mutation, build_custom_mutation
mutation_operator = build_custom_mutation({
    (0, 1)  : lambda x: Mutation.uniform_mutation(x, 1, -1, 1),
    (1, 2)  : lambda x: Mutation.uniform_mutation(x, 1, -8, 8),
    (2, 3)  : lambda x: Mutation.uniform_mutation(x, 1, -2, 2),
    (3, 4)  : lambda x: Mutation.gaussian_mutation(x, 1, 0, 0.25)
})

# Define the evaluate_individual function
from setup import evaluate_fitness as evaluate_isetndividual

# Define the fitness evaluator
from setup import fitness_evaluator as fe
def fitness_evaluator(**kwargs):
    return fe(evaluate_individual=evaluate_individual, **kwargs)

# Define the termination condition
termination_condition = lambda fitness_values : np.max(fitness_values) == 0
