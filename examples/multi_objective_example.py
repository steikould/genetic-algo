# An example script demonstrating a multi-objective optimization run.
import json
from ga_optimizer.src.services.optimization_service import run_optimization
from ga_optimizer.src.data.providers.synthetic import Inputs
from ga_optimizer.src.utils.serialization import CustomJSONEncoder

def run_multi_objective_optimization():
    """
    Runs a multi-objective optimization and prints the results.
    """
    print("Running multi-objective optimization example...")

    # 1. Initialize synthetic data
    inputs = Inputs()

    # 2. Define GA parameters for multi-objective
    ga_params = {
        "pop_size": 150,
        "ngen": 75,
        "cxpb": 0.7,
        "mutpb": 0.2,
        "sigma_value": 0.1,
        "optimization_mode": "hybrid"  # 'hybrid' for multi-objective
    }

    # 3. Run the optimization service
    # TODO: The optimization service needs to be adapted to handle multi-objective.
    # For now, this will run as a single-objective optimization.
    optimization_result = run_optimization(inputs, ga_params)

    # 4. Print the results
    result_json = json.dumps(
        optimization_result.model_dump(),
        cls=CustomJSONEncoder,
        indent=4
    )

    print("\n--- Multi-Objective Optimization Result (JSON) ---")
    print(result_json)

    # In a true multi-objective result, you would typically have a Pareto front
    # of non-dominated solutions instead of a single fitness value.
    print(f"\nFinal Fitness (as single objective for now): {optimization_result.performance_metrics.final_fitness}")

if __name__ == "__main__":
    run_multi_objective_optimization()
