# Contains the implementation of the Genetic Algorithm using the DEAP library.
import time
import numpy as np
from deap import base, creator, tools, algorithms
import random

from ....initialization.smart_seeding import smart_seed
from ....evaluation.fuel_evaluators import evaluate_ebitda
from .operators import mutate_and_normalize

# Create fitness class (single objective maximization)
# TODO: This should probably be done in a more centralized place
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def run_genetic_algorithm(inputs, pop_size=200, ngen=100, cxpb=0.6, mutpb=0.2, sigma_value=0.1):
    start_time = time.time()
    n_locations = inputs.n_locations

    # Smart seeding
    #TODO update center
    center = [1,1]
    init_qms = smart_seed(pop_size, center, inputs.S_transmix, inputs.S_target, n_locations)
    # TODO - [MXMC] - I'm here.
    # Create initial population
    pop = [creator.Individual(qm) for qm in init_qms]


    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: random.uniform(0, 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_locations)
    # The following line has an undefined `evaluate_individual` and `const`
    # toolbox.register("evaluate", evaluate_individual, inputs=inputs, const=const)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("select", tools.selTournament, tournsize=2)
    
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
