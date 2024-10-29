# from setup import *
from argparse import ArgumentParser
import importlib
import os


# Configure parser
parser = ArgumentParser(description='Run Bayesian Optimization')
parser.add_argument('--warmup_steps'        , type=int, default=5, help='Number of warmup steps')
parser.add_argument('--optimization_steps'   , type=int, default=5, help='Number of optimization steps')
parser.add_argument('--dataset'             , type=str, default='datasets/sinewave.parquet', help='Path to the dataset')
parser.add_argument('--batch_size'          , type=int, default=128, help='Batch size')
parser.add_argument('--setup_file'          , type=str, default='setups/setup-default.py', help='Path to the setup file')
parser.add_argument('--n_epochs'            , type=int, default=50, help='Number of epochs to train the model')
parser.add_argument('--tags'                , type=str, default=[], help='Tags for the run', nargs='+')



if __name__ == '__main__':
    # Parse the arguments
    args = parser.parse_args()

    # Import the setup file
    setup_file_path = os.path.abspath(args.setup_file)
    module_name = os.path.splitext(os.path.basename(setup_file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, setup_file_path)
    setup_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_module)

    # Get global variables defined in the setup file
    setup_module.iteration = 1
    setup_module.tags = args.tags
    setup_module.data_path = args.dataset
    setup_module.batch_size = args.batch_size
    setup_module.n_epochs = args.n_epochs

    # Tensor cores
    try:
        setup_module.torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    # Define BO optimizer obj
    optimizer = setup_module.BayesianOptimization(
        f            = setup_module.objective_function,
        pbounds      = setup_module.hyp_bounds,
        random_state =1
    )

    # Run the optimization
    optimizer.maximize(
        init_points = args.warmup_steps,
        n_iter      = args.optimization_steps
    )


    




