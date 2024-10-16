# GMOO-Project

This is the repository for the project of Global and Multi Objective Optimization course at the University of Trieste. 

The goal of the project is to implement a hyperparameter optimization algorithm using several techniques and compare them on a set of benchmark models and benchmark datasets.

The main techniques that we are going to use are:
- Bayesian Optimization (just for comparison)
- Genetic Algorithms
- Evolutionary Strategies
- Differential Evolution
- Particle Swarm Optimization


## Models

The benchmark models that we are going to use are:
- Random Forest
  - Number of trees $\in [5, 200]$
  - Maximum depth of the tree $\in [10, 100]$
  - Minimum samples to split a node $\in [2, 10]$
  - Minimum samples per leaf $\in [1, 10]$
  - Criterion $\in \{\text{gini}, \text{entropy}\}$

## Datasets

The benchmark datasets that we are going to use are:
- Iris


## Genetic Algorithms

The Genetic Algorithm is a population-based optimization algorithm that mimics the process of natural selection. Notice that we have mixed types of hyperparameters, so we need to define a proper encoding for the individuals.

For boolean hyperparameters, as the crietion of the Random Forest, we can use a binary encoding. For positive integer hyperparameters, as the number of trees, we can use a binary encoding. 
