from setups.setup_default import *
from bayes_opt import BayesianOptimization

iteration = 1
tags = []
data_path = ""
batch_size = 128
n_epochs = 100

hyp_bounds = {
    'n_layers': (1, 5),
    'hidden_size': (16, 128),
    'activation_idx': (0, len(ACTIVATIONS)-1),
    'dropout': (0, 0.5)
}

def objective_function(
    n_layers : int,
    hidden_size : int,
    activation_idx : int,
    dropout : float
):
    # Get global vars
    global iteration
    global tags
    global data_path
    global batch_size
    global n_epochs

    # Round hyperparameters
    n_layers    = int(round(n_layers    , 0))
    hidden_size = int(round(hidden_size , 0))
    activation_idx  = int(round(activation_idx  , 0))
    activation = ACTIVATIONS[activation_idx]

    # Create logger
    logger = Logger(
        project="GMOOP",
        name=f"BO-Iteration-{iteration}",
        tags=tags,
        config={
            'activation_function': activation_idx,
        }
    )

    # Get data
    D_train, D_val, D_test = load_data(data_path)
    train_loader, val_loader, test_loader = build_data_loaders(batch_size, D_train, D_val, D_test)

    # Instantiate the model
    model = MLP(
        input_size  = train_loader.dataset.tensors[0].shape[1],
        hidden_size = hidden_size,
        output_size = 1,
        n_layers    = n_layers,
        activation  = activation,
        dropout     = dropout,
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

    # Increment iteration
    iteration += 1

    # Return last validation loss
    return -last_val_loss
    

