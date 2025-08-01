import numpy as np
from scipy.optimize import minimize
from .genetic_algorithm import run_genetic_algorithm, evaluate_ebitda
from .models import (
    ExecutionContext,
    PerformanceMetrics,
    PopulationDynamics,
    StrategicInsights,
    TransferabilityMetadata,
    OptimizationResult,
)
from datetime import datetime
import json
import platform

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(CustomJSONEncoder, self).default(obj)

class Inputs:
    def __init__(self):
        self.n_locations = 3
        self.S_transmix = np.array([0.8, 0.1, 0.1])
        self.S_target = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.5, 0.4, 0.1]])
        self.diesel_price = 100
        self.transport_cost_matrix = np.array([10, 20, 30])
        self.storage_cost_per_unit = 5
        self.target_specs = self.S_target

def apply_local_optimizer(local_method, best_qm, inputs):
    """
    Apply local optimization to refine GA solution
    local_method: 'L-BFGS-B', 'SLSQP', etc.
    """

    # Constraint: allocations must sum to 1
    constraint = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}

    # Bounds: each allocation between 0 and 1
    bounds = [(0, 1) for _ in range(len(best_qm))]

    # Objective function (negate for minimization)
    def objective(x):
        return -evaluate_ebitda(x,
                                transmix_props=inputs.S_transmix,
                                target_specs=inputs.S_target,
                                diesel_price=inputs.diesel_price,
                                transport_cost_matrix=inputs.transport_cost_matrix,
                                storage_cost_per_unit=inputs.storage_cost_per_unit)[0]

    result = minimize(
        objective,
        best_qm,
        method=local_method,
        bounds=bounds,
        constraints=constraint
    )

    return result.x if result.success else best_qm

def main():
    inputs = Inputs()
    ga_params = {"pop_size": 200, "ngen": 100, "cxpb": 0.6, "mutpb": 0.2, "sigma_value": 0.1}

    ga_results = run_genetic_algorithm(inputs, **ga_params)
    best_qm = ga_results["best_qm"]
    max_ebitda = ga_results["final_fitness"]

    print("Best allocation from GA:", best_qm)
    print("Max EBITDA from GA:", max_ebitda)

    refined_qm = apply_local_optimizer('SLSQP', best_qm, inputs)

    refined_ebitda = evaluate_ebitda(refined_qm,
                                     transmix_props=inputs.S_transmix,
                                     target_specs=inputs.S_target,
                                     diesel_price=inputs.diesel_price,
                                     transport_cost_matrix=inputs.transport_cost_matrix,
                                     storage_cost_per_unit=inputs.storage_cost_per_unit)[0]

    print("Refined allocation from local optimization:", refined_qm)
    print("Refined EBITDA from local optimization:", refined_ebitda)

    # --- Populate Pydantic Models ---

    exec_context = ExecutionContext(
        algorithm_config={"name": "DEAP Genetic Algorithm", **ga_params},
        problem_characteristics={
            "diesel_price": inputs.diesel_price,
            "storage_cost_per_unit": inputs.storage_cost_per_unit,
        },
        hyperparameters=ga_params,
        hyperparameter_source="manual",
        computational_resources={
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        problem_domain="Fuel Blending Optimization",
        problem_size={"n_locations": inputs.n_locations},
    )

    perf_metrics_data = ga_results["performance_metrics"]
    perf_metrics = PerformanceMetrics(
        **perf_metrics_data,
        convergence_stability=-1.0, # Placeholder
        memory_peak=-1.0, # Placeholder
        efficiency_score=max_ebitda / perf_metrics_data["execution_time"] if perf_metrics_data["execution_time"] > 0 else 0,
        final_fitness=max_ebitda,
        solution_quality=max_ebitda, # Using raw fitness as quality
        robustness_score=-1.0, # Placeholder
        constraint_satisfaction=1.0, # By construction
        baseline_improvement=-1.0, # Placeholder
    )

    pop_dynamics_data = ga_results["population_dynamics"]
    pop_dynamics = PopulationDynamics(
        **pop_dynamics_data,
        crossover_success_rates=[-1.0], # Placeholder
        mutation_impact_scores=[-1.0], # Placeholder
        population_clustering_analysis={}, # Placeholder
    )

    strategic_insights = StrategicInsights(
        most_effective_operators={"crossover": 0.6, "mutation": 0.4}, # Placeholder
        optimal_parameter_ranges={"cxpb": (0.5, 0.8)}, # Placeholder
        critical_success_factors=["Smart Seeding", "Elitism"],
        failure_modes=["Premature Convergence"], # Placeholder
        recommended_strategies=["Hybrid GA with Local Search"],
        parameter_sensitivity_analysis={"sigma": 0.8}, # Placeholder
        transfer_learning_potential=0.5, # Placeholder
        causal_relationships={}, # Placeholder
        intervention_effects={}, # Placeholder
        counterfactual_scenarios=[], # Placeholder
    )

    transfer_metadata = TransferabilityMetadata(
        problem_similarity_features={"n_locations": 1.0}, # Placeholder
        generalization_boundaries={}, # Placeholder
        transfer_confidence=0.5, # Placeholder
        feature_importance_weights={}, # Placeholder
        context_adaptation_rules=[], # Placeholder
        transfer_learning_coefficients={}, # Placeholder
        cross_validation_scores=[],
        successful_transfers=[],
        failed_transfers=[],
    )

    optimization_result = OptimizationResult(
        execution_context=exec_context,
        performance_metrics=perf_metrics,
        population_dynamics=pop_dynamics,
        strategic_insights=strategic_insights,
        transferability_metadata=transfer_metadata,
    )

    # --- Print JSON Output ---
    result_json = json.dumps(
        optimization_result.model_dump(),
        cls=CustomJSONEncoder,
        indent=4
    )

    print("\n--- Optimization Result (JSON) ---")
    print(result_json)


if __name__ == "__main__":
    main()
