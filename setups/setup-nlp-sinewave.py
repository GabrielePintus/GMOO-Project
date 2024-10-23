from setups.setup_default import *



#
#       ALGORITHM COMPONENTS
#
Chromosome.set_domain({
    'n_layers': (1, 5),
    'hidden_size': (16, 128),
    'activation': list(range(len(ACTIVATIONS))),
    'dropout': (0, 0.5)
})

# Selection operator
selection_operator = lambda population, fitness_values: Selection.tournament_selection(population, fitness_values, 3)
# Cross-over pairing
crossover_pairing = Crossover.half_pairs
# Crossover operator
crossover_operator = Crossover.two_point_crossover
# Mutation operator
mutation_operator = build_custom_mutation({
    (0, 1)  : lambda x: Mutation.uniform_mutation(x, 1, -1, 1),
    (1, 2)  : lambda x: Mutation.uniform_mutation(x, 1, -8, 8),
    (2, 3)  : lambda x: Mutation.uniform_mutation(x, 1, -2, 2),
    (3, 4)  : lambda x: Mutation.gaussian_mutation(x, 1, 0, 0.15)
})
# Termination condition
termination_condition = lambda fitness_values : np.max(fitness_values) == 0