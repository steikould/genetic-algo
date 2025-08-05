# Provides the main service for orchestrating the optimization process.
import numpy as np
import platform
import json

from ..algorithms.genetic.deap_ga import run_genetic_algorithm
from ..algorithms.local.scipy_optimizers import apply_local_optimizer
from ..domain.models import (
    ExecutionContext,
    PerformanceMetrics,
    PopulationDynamics,
    StrategicInsights,
    TransferabilityMetadata,
    OptimizationResult,
)
from ..data.providers.synthetic import Inputs
from ..utils.serialization import CustomJSONEncoder
from ..evaluation.fuel_evaluators import evaluate_ebitda


def run_optimization(inputs, ga_params):
    """
    Runs the full optimization process, including GA and local refinement.
    """
    ga_results = run_genetic_algorithm(inputs, **ga_params)
    best_qm = ga_results["best_qm"]
    max_ebitda = ga_results["final_fitness"]

    refined_qm = apply_local_optimizer('SLSQP', best_qm, inputs)

    refined_ebitda = evaluate_ebitda(refined_qm,
                                     transmix_props=inputs.S_transmix,
                                     target_specs=inputs.S_target,
                                     diesel_price=inputs.diesel_price,
                                     transport_cost_matrix=inputs.transport_cost_matrix,
                                     storage_cost_per_unit=inputs.storage_cost_per_unit)[0]

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

    return optimization_result
