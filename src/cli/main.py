# The main entry point for the command-line interface (CLI).
import json
from ..services.optimization_service import run_optimization
from ..data.providers.synthetic import Inputs
from ..utils.serialization import CustomJSONEncoder

def main():
    """
    Main function to run the optimization and print results.
    """
    inputs = Inputs()
    ga_params = {"pop_size": 200, "ngen": 100, "cxpb": 0.6, "mutpb": 0.2, "sigma_value": 0.1}

    optimization_result = run_optimization(inputs, ga_params)

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
