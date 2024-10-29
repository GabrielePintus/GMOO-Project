from setups.setup_default import *
from bayes_opt import BayesianOptimization




#
#       DATA LOADING
#
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def load_data(path, seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
    ])
    
    # Load MNIST dataset
    full_train_dataset = datasets.MNIST(root=path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=path, train=False, download=True, transform=transform)

    # Split full_train_dataset into training and validation sets
    train_indices, val_indices = train_test_split(range(len(full_train_dataset)), test_size=0.2, random_state=seed)
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    # Extract X and y from the Subset datasets, converting to numpy arrays
    X_train = torch.stack([train_dataset[i][0].view(-1) for i in range(len(train_dataset))]).numpy()
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))]).numpy().reshape(-1, 1)
    
    X_val = torch.stack([val_dataset[i][0].view(-1) for i in range(len(val_dataset))]).numpy()
    y_val = torch.tensor([val_dataset[i][1] for i in range(len(val_dataset))]).numpy().reshape(-1, 1)

    X_test = test_dataset.data.reshape(-1, 28*28).numpy()
    y_test = test_dataset.targets.numpy().reshape(-1, 1)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def build_data_loaders(batch_size, D_train, D_val, D_test):
    # Create PyTorch tensors
    X_train , y_train   = torch.tensor(D_train[0], dtype=torch.float32), torch.tensor(D_train[1], dtype=torch.float32)
    X_val   , y_val     = torch.tensor(D_val[0], dtype=torch.float32), torch.tensor(D_val[1], dtype=torch.float32)
    X_test  , y_test    = torch.tensor(D_test[0], dtype=torch.float32), torch.tensor(D_test[1], dtype=torch.float32)

    # Create PyTorch datasets
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



#      
#      BAYESIAN OPTIMIZATION
#
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
    

