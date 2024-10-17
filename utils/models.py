from sklearn.ensemble import RandomForestClassifier
from torch import nn
from utils.Types import Chromosome

class RandomForest(RandomForestClassifier):
    def __init__(self, **kwargs):
        # The last position is the criterion
        # Map 0 to 'gini' and 1 to 'entropy'
        kwargs['criterion'] = ['gini', 'entropy'][kwargs['criterion']]
        super().__init__(**kwargs)
        
        
class MLP(nn.Module):
    def __init__(
        self,
        input_size  : int,
        hidden_size : int,
        output_size : int,
        activation  : nn.Module,
        dropout     : float = 0.1
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = [nn.ReLU(), nn.Tanh()][activation]
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        

        
if __name__ == '__main__':
    Chromosome.set_domain({ 
        'hidden_size': (10, 100),
        'activation': [nn.ReLU(), nn.Tanh()],
        'dropout': (0.1, 0.5),
    })
    
    params = {
        'hidden_size': 16,
        'activation': 0,
        'dropout': 0.2,
    }
    
    ch = Chromosome.list_to_dict(Chromosome.map, params)
    print(ch)
    
    