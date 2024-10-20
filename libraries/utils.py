from joblib import Parallel, delayed
from libraries.Types import Chromosome


def evaluate_multiple_fitness_parallel(evaluate_fitness, chromosomes, X_train, y_train, X_val, y_val):
    return Parallel(n_jobs=-1)(delayed(evaluate_fitness)(chromosome, X_train, y_train, X_val, y_val) for chromosome in chromosomes)

def evaluate_multiple_fitness(evaluate_fitness, chromosomes, X_train, y_train, X_val, y_val):
    return [evaluate_fitness(chromosome, X_train, y_train, X_val, y_val) for chromosome in chromosomes]

def initialize_population(population_size):
    return [Chromosome() for _ in range(population_size)]