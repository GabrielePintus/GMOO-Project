import random
import numpy as np

class Crossover:
    
    @staticmethod
    def one_point_crossover(parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(0, len(parent1))
        # Create offspring
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    
    @staticmethod
    def two_point_crossover(parent1, parent2):
        # Select two random crossover points
        crossover_points = sorted(random.sample(range(len(parent1)), 2))
        # Create offspring
        child1 = parent1[:crossover_points[0]] + parent2[crossover_points[0]:crossover_points[1]] + parent1[crossover_points[1]:]
        child2 = parent2[:crossover_points[0]] + parent1[crossover_points[0]:crossover_points[1]] + parent2[crossover_points[1]:]
        return child1, child2
    
    @staticmethod
    def uniform_crossover(parent1, parent2, p=0.5):
        # Create offspring
        binomial_vector = np.random.binomial(1, p, len(parent1))
        child1 = [parent1[i] if binomial_vector[i] < p else parent2[i] for i in range(len(parent1))]
        child2 = [parent2[i] if binomial_vector[i] < p else parent1[i] for i in range(len(parent2))]
        return child1, child2
    
    @staticmethod
    def intermediate_recombination(parent1, parent2):
        alpha = random.uniform(0, 1)
        # Create offspring
        child1 = [alpha * parent1[i] + (1 - alpha) * parent2[i] for i in range(len(parent1))]
        child2 = [alpha * parent2[i] + (1 - alpha) * parent1[i] for i in range(len(parent2))]
        return child1, child2
    
    @staticmethod
    def line_recombination(parent1, parent2, k=2):
        alpha = random.uniform(-k, 1+k)
        # Create offspring
        child1 = [alpha * parent1[i] + (1 - alpha) * parent2[i] for i in range(len(parent1))]
        child2 = [alpha * parent2[i] + (1 - alpha) * parent1[i] for i in range(len(parent2))]
        return child1, child2
    
    @staticmethod
    def pmx(parent1, parent2):
        length = len(parent1)
        p1, p2 = sorted(random.sample(range(length), 2))

        offspring1, offspring2 = [None] * length, [None] * length
        offspring1[p1:p2], offspring2[p1:p2] = parent1[p1:p2], parent2[p1:p2]

        def fill_offspring(offspring, parent, p1, p2):
            for i in range(p1, p2):
                if parent[i] not in offspring:
                    pos = i
                    while offspring[pos] is not None:
                        pos = parent.index(offspring[pos])
                    offspring[pos] = parent[i]
            for i, gene in enumerate(offspring):
                if gene is None:
                    offspring[i] = parent[i]
        
        fill_offspring(offspring1, parent2, p1, p2)
        fill_offspring(offspring2, parent1, p1, p2)

        return offspring1, offspring2
    
    # Crossover pairings
    @staticmethod
    def random_pairs(population, k):
        possible_pairs = [(i, j) for i in range(len(population)) for j in range(i+1, len(population))]
        assert k <= len(possible_pairs), "k must be less than or equal to the number of possible pairs"
        idxs = random.sample(possible_pairs, k)
        return [(population[i], population[j]) for i, j in idxs]

    @staticmethod
    def all_pairs(population):
        idxs = [(i, j) for i in range(len(population)) for j in range(i+1, len(population))]
        return [(population[i], population[j]) for i, j in idxs]
    
    @staticmethod
    def half_pairs(population):
        half = len(population) // 2
        return [(population[i], population[i+half]) for i in range(half)]


class Mutation:
    
    @staticmethod
    def bit_flip_mutation(individual, mutation_rate=0.5):
        mutated_individual = [1 - gene if random.random() < mutation_rate else gene for gene in individual]
        return mutated_individual
    
    @staticmethod
    def random_reset_mutation(individual, mutation_rate=0.5, low=0, high=1):
        mutated_individual = [random.randint(low, high) if random.random() < mutation_rate else gene for gene in individual]
        return mutated_individual
        
    @staticmethod
    def gaussian_mutation(individual, mutation_rate=0.5, mean=0, std=1):
        mutated_individual = [gene + random.gauss(mean, std) if random.random() < mutation_rate else gene for gene in individual]
        return mutated_individual
    
    @staticmethod
    def uniform_mutation(individual, mutation_rate=0.5, low=0, high=1):
        mutated_individual = [gene + random.randint(low, high) if random.random() < mutation_rate else gene for gene in individual]
        return mutated_individual
    
    @staticmethod
    def transition_mutation(individual, mutation_rate=0.5, transition_matrix=None, outcomes=None):
        assert transition_matrix is not None, "transition_matrix must be provided"
        assert outcomes is not None, "outcomes must be provided"
        for i, gene in enumerate(individual):
            probabilities = transition_matrix[gene]
            print('outcomes:', outcomes)
            print('probabilities:', probabilities)
            sample = random.choices(outcomes, weights=probabilities, k=1)
            individual[i] = sample[0] if random.random() < mutation_rate else gene
        return individual
    
    

def build_custom_crossover(map):
    """
    Build a custom crossover operator from a map of indexes to functions.
    For example:
    {
        0:2 -> Crossover.one_point_crossover,
        2:5 -> Crossover.uniform_crossover,
    }
    """
    def custom_crossover(parent1, parent2):
        length = len(parent1)
        child1, child2 = [None] * length, [None] * length
        for _slice, crossover in map.items():
            child1[_slice], child2[_slice] = crossover(parent1[_slice], parent2[_slice])
        return child1, child2
    return custom_crossover


def build_custom_mutation(map):
    """
    Build a custom mutation operator from a map of indexes to functions.
    For example:
    {
        0:2 -> Mutation.bit_flip_mutation,
        2:5 -> Mutation.gaussian_mutation,
    }
    """
    def custom_mutation(individual):
        length = len(individual)
        mutated_individual = [None] * length
        for _slice, mutation in map.items():
            _slice = slice(*_slice)
            mutated_individual[_slice] = mutation(individual[_slice])
        return mutated_individual
    return custom_mutation



class Selection:
    
    @staticmethod
    def tournament_selection(population, fitness_values, tournament_size):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(list(enumerate(fitness_values)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            selected.append(population[winner[0]])
        return selected
    
    @staticmethod
    def roulette_wheel_selection(population, fitness_values):
        total_fitness = sum(fitness_values)
        probabilities = [fitness / total_fitness for fitness in fitness_values]
        selected = []
        for _ in range(len(population)):
            r = random.random()
            acc = 0
            for i, p in enumerate(probabilities):
                acc += p
                if r < acc:
                    selected.append(population[i])
                    break
        return selected
    
    @staticmethod
    def rank_selection(population, fitness_values):
        ranks = sorted(range(len(fitness_values)), key=lambda x: fitness_values[x])
        probabilities = [r / (len(ranks) - 1) for r in range(len(ranks))]
        selected = []
        for _ in range(len(population)):
            r = random.random()
            acc = 0
            for i, p in enumerate(probabilities):
                acc += p
                if r < acc:
                    selected.append(population[ranks[i]])
                    break
        return selected
    
    @staticmethod
    def elitism_selection(population, fitness_values, elite_size):
        ranks = sorted(range(len(fitness_values)), key=lambda x: fitness_values[x])
        return [population[i] for i in ranks[:elite_size]]