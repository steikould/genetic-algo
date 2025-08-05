# Contains smart initialization and seeding strategies for the population.
import numpy as np

def smart_seed(pop_size, center, S_transmix, S_target, n_locations, epsilon=1e-6, noise=0.02):
    """
    Create smart initialization function
    """
    ####################################################################################################
    # Option 1: detailed, fine-grained weighting based on every single data point's difference
    delta_weights = 1 / np.clip(np.abs(np.array(S_transmix[:n_locations]) - S_target), 1e-6, None)
    delta_weights /= np.sum(delta_weights)
    # Option 2: while the second provides a coarser, summary weighting based on the average behaviors of
    # rows versus an overall average.
    # Calculate inverse-delta weights: higher weight = closer to target
    # delta_weights = 1.0 / (np.abs(S_target.mean(axis=1) - S_transmix.mean()) + epsilon)
    ####################################################################################################

    # Generate population using Dirichlet distribution
    init_qms = []
    population=[center]
    for _ in range(pop_size - 1):
        alpha = delta_weights * 20  # Concentration parameter
        sample = np.random.dirichlet(alpha)
        sample = np.clip(sample + np.random.normal(0, noise, n_locations), 0, 1)
        sample /= np.sum(sample)
        population.append(sample.to_list())

    return population
