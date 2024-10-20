from setup import *


# Tensor cores
torch.set_float32_matmul_precision('high')


if __name__ == '__main__':
    # Load the data
    X_train, y_train, X_val, y_val  = load_data('datasets/sinewave.parquet')
    BATCH_SIZE = 128
    train_loader, val_loader        = build_data_loaders(BATCH_SIZE, X_train, y_train, X_val, y_val)

    # ES params
    pop_size = 4
    max_generations = 2
    total_fitness_evaluations = pop_size * max_generations

    # Loggers
    wandb_loggers = [ WandbLogger(
        project="GMOOP",
        name=f"run-{i}",
    ) for i in range(total_fitness_evaluations)]

    # Algorithm object
    ga = GeneticAlgorithm(
        population_initializer  = population_initializer,
        fitness_evaluator       = fitness_evaluator,
        selection_operator      = lambda pop, fitness : Selection.tournament_selection(pop , fitness, 2),
        crossover_pairing       = Crossover.half_pairs,
        crossover_operator      = Crossover.uniform_crossover,
        mutation_operator       = custom_mutation,
        termination_condition   = lambda fitness_values : np.max(fitness_values) == 0
    )

    # Run the algorithm
    ga.run(
        (train_loader, val_loader),
        pop_size,
        max_generations,
        ['sine-wave']
    )