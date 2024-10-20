from setup import *


if __name__ == '__main__':
    # Load the data
    X_train, y_train, X_val, y_val  = load_data('datasets/sinewave.parquet')
    BATCH_SIZE = 128
    train_loader, val_loader        = build_data_loaders(BATCH_SIZE, X_train, y_train, X_val, y_val)

    # Tensor cores
    torch.set_float32_matmul_precision('high')

    # Logger
    wandb_logger = WandbLogger(
        project="GMOOP",
        name="prova-nome-run",
    )

    # Training
    chromosome = Chromosome(
        value=[16, 1, 0.1]
    )
    print('chromosome:', chromosome)
    evaluate_fitness(
        chromosome=chromosome,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=20,
        logger = wandb_logger
    )