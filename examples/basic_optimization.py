# An example script demonstrating a basic optimization run.
import json
from ga_optimizer.src.services.optimization_service import run_optimization
from ga_optimizer.src.data.providers.synthetic import Inputs
from ga_optimizer.src.utils.serialization import CustomJSONEncoder

def run_basic_optimization():
    """
    Runs a basic optimization and prints the results.
    """
    print("Running basic optimization example...")

    # 1. Initialize synthetic data
    inputs = Inputs()

    # 2. Define GA parameters
    ga_params = {
        "pop_size": 100,
        "ngen": 50,
        "cxpb": 0.7,
        "mutpb": 0.2,
        "sigma_value": 0.1
    }

    # 3. Run the optimization service
    optimization_result = run_optimization(inputs, ga_params)

    # 4. Print the results
    result_json = json.dumps(
        optimization_result.model_dump(),
        cls=CustomJSONEncoder,
        indent=4
    )

    print("\n--- Optimization Result (JSON) ---")
    print(result_json)

    # You can access specific results like this:
    print(f"\nFinal Fitness: {optimization_result.performance_metrics.final_fitness}")

if __name__ == "__main__":
    run_basic_optimization()
