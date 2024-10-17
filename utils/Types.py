import random

class ModelAdaptor:
    
    def __init__(self, model_class, params_map):
        self.model_class = model_class
        self.params_map = params_map

    def build(self, hyperparams):
        print(hyperparams)
        if isinstance(hyperparams, list):
            hyperparams = {key:h for key, h in zip(self.params_map.keys(), hyperparams)}
        print(hyperparams)
        
        casted_hyperparams = {}
        for key, value in hyperparams.items():
            hyperparam = self.params_map.get(key, None) if isinstance(self.params_map.get(key, None), dict) else None
            if hyperparam is not None:
                try:
                    casted = hyperparam[value]
                except KeyError:
                    raise ValueError(f'Value "{value}" not specificied in the casting map for hyperparameter "{key}"')
            else:
                casted = value
            casted_hyperparams[key] = casted
        return self.model_class(**casted_hyperparams)






class Chromosome:
    
    domain : dict = None
    map    : dict = None
    
    @staticmethod
    def set_domain(domain):
        Chromosome.domain = domain
        Chromosome.map = { i:key for i, key in enumerate(Chromosome.domain.keys()) }
    
    @staticmethod
    def list_to_dict(hyperparams):
        hyperparams = {Chromosome.map[i]:h for i, h in enumerate(hyperparams)}
        # Cast hyperparameters to their original type
        for key, value in Chromosome.domain.items():
            if isinstance(value, list):
                # This means we are defining the domain as a list of possible values
                hyperparams[key] = value[hyperparams[key]]
        return hyperparams
    
    @staticmethod
    def random_initialization():
        random_chromosome = []
        for key, value in Chromosome.domain.items():
            if isinstance(value, tuple):
                # This means we are defining the domain as an interval
                random_chromosome.append(random.randint(value[0], value[1]))
            elif isinstance(value, list):
                # This means we are defining the domain as a list of possible values
                # random_chromosome.append(random.choice(value))
                random_chromosome.append(value.index(random.choice(value)))
        return random_chromosome
    
    def __init__(self, value=None):
        """
            domain: dict = {
                'n_estimators': (10, 100),
                'criterion': ['gini', 'entropy'],
            }
        """
        assert Chromosome.domain is not None, 'Chromosome domain must be defined before creating instances'
        self.value = value if value is not None else Chromosome.random_initialization()
        Chromosome.map = { i:key for i, key in enumerate(Chromosome.domain.keys()) }
    
    def check_bounds(self):
        for i, (key, value) in enumerate(self.domain.items()):
            if isinstance(value, tuple):
                # This means we are defining the domain as an interval
                self.value[i] = max(value[0],min(value[1], self.value[i]))
            elif isinstance(value, list):
                # This means we are defining the domain as a list of possible values
                if self.value[i] not in value:
                    raise ValueError(f'Value "{self.value[i]}" not in the domain of hyperparameter "{key}"')
        
    def mutate(self, operator):
        self.value = operator(self.value)
        self.check_bounds()
        return self
        
    def cross(self, other, operator):
        child1, child2 = operator(self.value, other.value)
        child1, child2 = Chromosome(child1), Chromosome(child2)
        child1.check_bounds()
        child2.check_bounds()
        return child1, child2
    
    # Override cast to dict
    def to_dict(self):
        return Chromosome.list_to_dict(self.value)
    
    def __str__(self):
        return str(self.value)
        
    def __len__(self):
        return len(self.value)





if __name__ == '__main__':
    Chromosome.set_domain({ 
        'n_estimators': (10, 100),
        'criterion': [min, max],
    })
    
    ch = Chromosome()
    print(ch.map)
    print(ch.value)
    print(ch.to_dict())
    