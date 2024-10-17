from tqdm.auto import tqdm
from utils.Types import Chromosome


class GeneticAlgorithm:
    
    def __init__(
        self,
        population_initializer,
        fitness_evaluator,
        selection_operator,
        crossover_pairing,
        crossover_operator,
        mutation_operator,
        termination_condition
    ):
        self.population_initializer = population_initializer
        self.fitness_evaluator = fitness_evaluator
        self.selection_operator = selection_operator
        self.crossover_pairing = crossover_pairing
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.termination_condition = termination_condition
        
    def run(
        self,
        data,
        population_size,
        max_generations
    ):
        # Initialize population
        population = self.population_initializer(population_size)
        
        # Create "history" lists to track the evolution of the best chromosome
        population_history = [population]
        fitness_history = []
        
        # Evolution loop
        best_chromosome = None
        best_fitness = float('-inf')
        progress_bar = tqdm(range(max_generations), desc='Generations')
        for generation in progress_bar:
            progress_bar.set_description(f'Generation: {generation}')
            # Evaluate fitness
            fitness = self.fitness_evaluator(population, data)
            fitness_history.append(fitness)
            
            # Select best chromosome
            best_index = fitness.index(max(fitness))
            if fitness[best_index] >= best_fitness:
                best_fitness = fitness[best_index]
                best_chromosome = population[best_index]
            
            # Check termination condition
            if self.termination_condition(fitness):
                break
            
            # Select next generation
            selected = self.selection_operator(population, fitness)
            
            # Apply crossover
            children = []
            for pair in self.crossover_pairing(selected):
                # offspring = self.crossover_operator(*pair)
                offspring = pair[0].cross(pair[1], self.crossover_operator)
                children.extend(offspring)
                
            # Apply mutation
            # mutated_children = [self.mutation_operator(child) for child in children]
            mutated_children = [child.mutate(self.mutation_operator) for child in children]
            
            # Store the current population and move to the next generation
            population = mutated_children
            population_history.append(population)
            
            # Update progress bar
            progress_bar.set_postfix({
                'best_fitness': best_fitness
            })
        
        self.best_chromosome = best_chromosome
        self.best_fitness = best_fitness
        self.history = {
            'population': population_history,
            'fitness': fitness_history
        }
        