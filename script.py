# from setup import *
from argparse import ArgumentParser
import importlib
import os


# Configure parser
parser = ArgumentParser(description='Run the Genetic Algorithm')
parser.add_argument('--pop_size'        , type=int, default=5                           , help='Population size')
parser.add_argument('--max_generations' , type=int, default=20                          , help='Maximum number of generations')
parser.add_argument('--dataset'         , type=str, default='datasets/sinewave.parquet' , help='Path to the dataset')
parser.add_argument('--batch_size'      , type=int, default=128                         , help='Batch size')
parser.add_argument('--n_epochs'        , type=int, default=50                          , help='Number of epochs to train the model')
parser.add_argument('--tags'            , type=str, default=[]                          , help='Tags for the run', nargs='+')
parser.add_argument('--setup_file'      , type=str, default='setups/setup-default.py'   , help='Path to the setup file')



if __name__ == '__main__':
    # Parse the arguments
    args = parser.parse_args()

    # Import the setup file
    setup_file_path = os.path.abspath(args.setup_file)
    module_name = os.path.splitext(os.path.basename(setup_file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, setup_file_path)
    setup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_module)

    # Tensor cores
    try:
        setup_module.torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # Load the data
    batch_size                            = args.batch_size
    D_train, D_val, D_test                = setup_module.load_data(args.dataset)
    train_loader, val_loader, _           = setup_module.build_data_loaders(batch_size, D_train, D_val, D_test)

    # ES params
    pop_size                    = args.pop_size
    max_generations             = args.max_generations
    total_fitness_evaluations   = pop_size * max_generations

    # Algorithm object
    ga = setup_module.GeneticAlgorithm(
        population_initializer  = setup_module.population_initializer,
        evaluate_population     = setup_module.evaluate_population,
        selection_operator      = setup_module.selection_operator,
        crossover_pairing       = setup_module.crossover_pairing,
        crossover_operator      = setup_module.crossover_operator,
        mutation_operator       = setup_module.mutation_operator,
        termination_condition   = setup_module.termination_condition
    )

    # Run the algorithm
    ga.run(
        (train_loader, val_loader),
        pop_size,
        max_generations,
        run_tags=args.tags,
        n_epochs=args.n_epochs,
        Logger = setup_module.Logger
    )