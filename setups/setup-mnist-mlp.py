from setups.setup_default import *

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
#       ALGORITHM COMPONENTS
#
Chromosome.set_domain({
    'n_layers': (1, 5),
    'hidden_size': (16, 128),
    'activation': list(range(len(ACTIVATIONS))),
    'dropout': (0, 0.5)
})
transition_matrix = np.loadtxt('setups/transition_probabilities.csv', delimiter=',')

# Selection operator
selection_operator = lambda population, fitness_values: Selection.tournament_selection(population, fitness_values, 3)
# Cross-over pairing
crossover_pairing = Crossover.half_pairs
# Crossover operator
crossover_operator = Crossover.two_point_crossover
# Mutation operator
mutation_operator = build_custom_mutation({
    (0, 1)  : lambda x: Mutation.uniform_mutation(x, 1, -2, 2),
    (1, 2)  : lambda x: Mutation.uniform_mutation(x, 1, -16, 16),
    (2, 3)  : lambda x: Mutation.transition_mutation(x, 1, transition_matrix, list(range(len(ACTIVATIONS))) ),
    (3, 4)  : lambda x: Mutation.gaussian_mutation(x, 1, 0, 0.15)
})
# Termination condition
termination_condition = lambda fitness_values : np.max(fitness_values) == 0