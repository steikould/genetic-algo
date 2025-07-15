import numpy as np
from scipy.optimize import minimize
from .genetic_algorithm import run_genetic_algorithm, evaluate_ebitda

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
    best_qm, max_ebitda = run_genetic_algorithm(inputs)

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

if __name__ == "__main__":
    main()
