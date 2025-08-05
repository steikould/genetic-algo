# Defines the genetic operators like crossover, mutation, and selection.
import numpy as np
from deap import tools

def mutate_and_normalize(individual, mu=0, sigma=0.1, indpb=0.2):
    """Apply Gaussian mutation then renormalize to sum=1"""
    tools.mutGaussian(individual, mu, sigma, indpb)
    individual[:] = np.clip(individual, 0, 1)  # Ensure non-negative
    individual[:] = individual / np.sum(individual)  # Normalize
    return individual,
