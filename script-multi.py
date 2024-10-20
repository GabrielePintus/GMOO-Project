from setup import *


if __name__ == '__main__':
    # Load the data
    X_train, y_train, X_val, y_val  = load_data('datasets/sinewave.parquet')
    BATCH_SIZE = 128
    train_loader, val_loader        = build_data_loaders(BATCH_SIZE, X_train, y_train, X_val, y_val)

    # Tensor cores
    torch.set_float32_matmul_precision('high')

    # GA params
    pop_size = 3

    # Logger(s)
    wandb_loggers = [ WandbLogger(
        project="GMOOP",
        name=f"run-{i}",
    ) for i in range(pop_size) ]

    # Training
    chromosomes = [ Chromosome() for _ in range(pop_size) ]
    fitnesses = fitness_evaluator(
        chromosomes,
        (train_loader, val_loader),
        n_epochs=10,
        loggers=wandb_loggers
    )
    