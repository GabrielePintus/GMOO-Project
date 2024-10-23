# GMOO-Project

This is the repository for the project of Global and Multi Objective Optimization course at the University of Trieste. 

The goal of the project is to implement a hyperparameter optimization algorithm using evolution strategies. The method is then tested on several models and benchmark datasets.


## Models

The benchmark models that we are going to use are:
- n-layers fully connected neural network


## Datasets

The benchmark datasets that we are going to use are:
- Sinewave (custom toy dataset)
- MNIST


## Genetic Algorithms

When dealing with genetic algorithms there are several components that need to be defined. We will test several alternatives for each of these components.
To sum up the different versions of the genetic algorithm we define:
- GA-Basic
  - Population size: 5
  - Number of generations: 20
  - Selection: Tournament selection with tournament size 3
  - Crossover pairing: half-pairs
  - Crossover: Double point crossover
  - Mutation: Uniform mutation for discrete hyperparameters and Gaussian mutation for continuous hyperparameters

### The mutation operator

The problems with the mutation operator arise when the search space is not "uniform" in the sense that each direction has a different domain. For example, suppose we want to tune the number of layer and the dropout rate. The first is a positive integer, while the second is a real number. Using the same mutation operator for both hyperparameters would not make much sense. Moreover, even when considering same type variables, for example number of layers and number of neurons per layer, the scale of the two variables is different, with the first being usually much smaller than the second. Using the same mutation operator with the same variability would not make much sense either.

How to overcome this problem? One possible solution is to use a different mutation operator for each different class of hyperparameters.
For example:
- number of layers: n + Uniform(-1, 1)
- number of neurons per layer: n + Uniform(-10, 10)
- dropout rate: n + Gaussian(0, 0.25)
- activation function: ?


#### The problem with the activation function

Because the activation function variable is categorical, how to mutate it is not clear. One trivial solution will be to enumerate all the possible activation functions and use a uniform mutation.
- 0: ReLU
- 1: Leaky ReLU
- 2: ELU
- 3: GELU
- 4: Sigmoid
- 5: Tanh

This solution, however, is not very elegant. We are indirectly defining a distance function between activation functions, without any meaningful reason. Why is the distance between ReLU and Leaky ReLU the same as the distance between GELU and Sigmoid?
Additionally, any permutation of this enumeration would give different results.

One better way to define the mutation operator, is to directly define a distance function between activation functions, and then induce a probability distribution. Then, we can sample from this distribution to get the new activation function. 

In order to do so we first build a hierarchy of activation functions and represent it as a tree. Our choice is the following:
```
Activation Function
├── Unbounded
│   ├── Straight
│   │   ├── relu
│   │   └── leakyrelu
│   └── Round
│       ├── elu
│       └── gelu
└── Bounded
    ├── tanh
    └── sigmoid
```

This way, we encode the differences between activation functions in the distance we define.
The distance is defined as the number of hops between two nodes in the tree.
```math
d_{\text{hops}}(a, b) = \text{number of hops between a and b}
```
An alternative could be to define this distance as the logarithm of the number of hops. This way the distance between two nodes is more "uniformly" distributed, allowing for more exploration.

Example: probability of mutation from ReLU to every other activation function
| Activation Function | Probability with hops (%) | Probability with log_2(hops) (%) |
|----------------------|-----------------------|----------------------------|
| Leaky ReLU           | 57.1                  | 35.7                       |
| ELU                  | 14.3                  | 17.9                       |
| GELU                 | 14.3                  | 17.9                       |
| Sigmoid              | 7.1                   | 14.3                       |
| Tanh                 | 7.1                   | 14.3                       |

By adjusting the base of the logarithm we can control the exploration-exploitation trade-off. The higher the base, the more exploration we have. Notice that this value must be greater than 1, otherwise the distance is always 0. The extreme cases are:

```math
\begin{align*}

\lim_{b \to 1} p_b(a,b) &= \begin{cases}
1 & \text{if } a \text{ and } b \text{ are in the same branch} \\
0 & \text{otherwise}
\end{cases} \\

\lim_{b \to \infty} p_b(a,b) &= \frac{1}{\text{\# Activation Functions}} \quad \forall a, b

\end{align*}
```
