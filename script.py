# from setup import *
from argparse import ArgumentParser
import importlib


# Configure parser
parser = ArgumentParser(description='Run the Genetic Algorithm')
parser.add_argument('--pop_size', type=int, default=5, help='Population size')
parser.add_argument('--max_generations', type=int, default=20, help='Maximum number of generations')
parser.add_argument('--dataset', type=str, default='sinewave.parquet', help='Path to the dataset')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train the model')
parser.add_argument('--tags', type=str, nargs='+', default=[], help='Tags for the run')
parser.add_argument('--setup_file', type=str, default='setup.py', help='Path to the setup file')
args = parser.parse_args()


if __name__ == '__main__':
    # Import the setup file
    setup = importlib.import_module(args.setup_file.replace('.py', ''))
    # globals().update(vars(setup_module))

    # Tensor cores
    setup.torch.set_float32_matmul_precision('high')

    # Load the data
    data_path = args.dataset
    batch_size = args.batch_size
    X_train, y_train, X_val, y_val  = setup.load_data(data_path)
    train_loader, val_loader        = setup.build_data_loaders(batch_size, X_train, y_train, X_val, y_val)

    # ES params
    pop_size = args.pop_size
    max_generations = args.max_generations
    total_fitness_evaluations = pop_size * max_generations

    # Loggers
    wandb_loggers = [ setup.WandbLogger(
        project="GMOOP",
        name=f"run-{i}",
    ) for i in range(total_fitness_evaluations)]

    # Algorithm object
    ga = setup.GeneticAlgorithm(
        population_initializer  = setup.population_initializer,
        fitness_evaluator       = setup.fitness_evaluator,
        selection_operator      = lambda pop, fitness : setup.Selection.tournament_selection(pop , fitness, 3),
        crossover_pairing       = setup.Crossover.half_pairs,
        crossover_operator      = setup.Crossover.uniform_crossover,
        mutation_operator       = setup.custom_mutation,
        termination_condition   = lambda fitness_values : setup.np.max(fitness_values) == 0
    )

    # Run the algorithm
    ga.run(
        (train_loader, val_loader),
        pop_size,
        max_generations,
        run_tags=args.tags,
        n_epochs=args.n_epochs
    )