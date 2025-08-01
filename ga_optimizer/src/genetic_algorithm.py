import numpy as np
from deap import base, creator, tools, algorithms
import time

# Create fitness class (single objective maximization)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize toolbox
toolbox = base.Toolbox()


# Create smart initialization function
def smart_seed(pop_size, S_transmix, S_target, n_locations, epsilon=1e-6):
    """
    Create smart initialization function
    """
    # Calculate inverse-delta weights: higher weight = closer to target
    delta_weights = 1.0 / (np.abs(S_target.mean(axis=1) - S_transmix.mean()) + epsilon)

    # Generate population using Dirichlet distribution
    init_qms = []
    for _ in range(pop_size):
        alpha = delta_weights * 20  # Concentration parameter
        sample = np.random.dirichlet(alpha)
        init_qms.append(sample)

    return init_qms


def evaluate_ebitda(individual, transmix_props, target_specs, diesel_price, transport_cost_matrix, storage_cost_per_unit):
    """
    individual: split ratio allocation across locations [0.3, 0.4, 0.3]
    Returns: (ebitda_value,) - tuple with single objective
    """
    from .utils import calculate_swell_volume, calculate_blending_cost

    allocation = np.array(individual)

    total_ebitda = 0
    for i, qty in enumerate(allocation):
        # Calculate revenue from blending at location i
        swell_volume = calculate_swell_volume(qty, transmix_props, target_specs[i])
        revenue = diesel_price * swell_volume * qty

        # Calculate costs (transportation, blending, storage)
        transport_cost = transport_cost_matrix[i] * qty
        blending_cost = calculate_blending_cost(qty, target_specs[i])
        storage_cost = storage_cost_per_unit * qty

        location_ebitda = revenue - transport_cost - blending_cost - storage_cost
        total_ebitda += location_ebitda

    return (total_ebitda,)


def mutate_and_normalize(individual, mu=0, sigma=0.1, indpb=0.2):
    """Apply Gaussian mutation then renormalize to sum=1"""
    tools.mutGaussian(individual, mu, sigma, indpb)
    individual[:] = np.clip(individual, 0, 1)  # Ensure non-negative
    individual[:] = individual / np.sum(individual)  # Normalize
    return individual,


def run_genetic_algorithm(inputs, pop_size=200, ngen=100, cxpb=0.6, mutpb=0.2, sigma_value=0.1):
    start_time = time.time()
    n_locations = inputs.n_locations

    # Smart seeding
    init_qms = smart_seed(pop_size, inputs.S_transmix, inputs.S_target, n_locations)

    # Create initial population
    pop = [creator.Individual(qm) for qm in init_qms]

    # Register evaluation function
    toolbox.register("evaluate", evaluate_ebitda,
                     transmix_props=inputs.S_transmix,
                     target_specs=inputs.S_target,
                     diesel_price=inputs.diesel_price,
                     transport_cost_matrix=inputs.transport_cost_matrix,
                     storage_cost_per_unit=inputs.storage_cost_per_unit)

    # Register operators
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_and_normalize, sigma=sigma_value, indpb=0.2)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    evaluations_count = len(pop)

    # --- Data Collection Initialization ---
    fitness_trajectory = []
    diversity_trajectory = []
    population_snapshots = [
        {"generation": 0, "snapshot": [list(ind) for ind in pop]}
    ]
    breakthrough_generations = []


    # Evolution loop with stagnation detection
    stagnation_counter = 0
    last_best_value = None
    max_stagnation = 20
    converged_generation = ngen

    for generation in range(ngen):
        # --- Data Collection ---
        current_best_ind = tools.selBest(pop, k=1)[0]
        current_best_fitness = current_best_ind.fitness.values[0]
        fitness_trajectory.append(current_best_fitness)

        all_fitnesses = [ind.fitness.values[0] for ind in pop]
        diversity_trajectory.append(np.std(all_fitnesses))

        # Create offspring
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)

        # Evaluate offspring
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        evaluations_count += len(offspring)

        # Elitism: keep best 5 individuals
        elite = tools.selBest(pop, k=5)
        selected = toolbox.select(offspring, k=len(pop) - len(elite))
        pop = elite + selected

        # Check for stagnation
        current_best = tools.selBest(pop, k=1)[0]
        current_best_value = current_best.fitness.values[0]

        if last_best_value is None or current_best_value > last_best_value * 1.001: # 0.1% improvement
            if last_best_value is not None and current_best_value > last_best_value * 1.05: # 5% improvement
                breakthrough_generations.append(generation)
            last_best_value = current_best_value
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Early stopping
        if stagnation_counter >= max_stagnation:
            print(f"Converged at generation {generation}")
            converged_generation = generation
            break

    # --- Final Data Collection ---
    if generation == ngen - 1:
        # Also capture final population if loop completes
        population_snapshots.append({"generation": ngen - 1, "snapshot": [list(ind) for ind in pop]})

    # Capture a mid-point snapshot if it wasn't already captured
    mid_gen_snapshot_exists = any(d['generation'] == int(ngen/2) for d in population_snapshots)
    if not mid_gen_snapshot_exists and int(ngen/2) < generation:
        population_snapshots.append({"generation": int(ngen/2), "snapshot": [list(ind) for ind in pop]})


    end_time = time.time()

    # Get best solution
    best = tools.selBest(pop, k=1)[0]
    best_qm = np.clip(best, 0, 1)
    best_qm = best_qm / np.sum(best_qm)
    final_fitness = best.fitness.values[0]

    results = {
        "best_qm": best_qm,
        "final_fitness": final_fitness,
        "performance_metrics": {
            "fitness_trajectory": fitness_trajectory,
            "diversity_trajectory": diversity_trajectory,
            "convergence_rate": converged_generation,
            "execution_time": end_time - start_time,
            "evaluations_count": evaluations_count,
        },
        "population_dynamics": {
             "population_snapshots": population_snapshots,
             "genetic_diversity_evolution": diversity_trajectory,
             "elite_preservation_rate": [5 / pop_size] * ngen,
             "selection_pressure_evolution": [3] * ngen, # tournsize
             "population_stagnation_periods": [(converged_generation - max_stagnation, converged_generation)] if converged_generation < ngen else [],
             "breakthrough_generations": breakthrough_generations,
        }
    }

    return results
