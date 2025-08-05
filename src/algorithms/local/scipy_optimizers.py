# Contains local optimization algorithms using the SciPy library.
import numpy as np
from scipy.optimize import minimize
from ....evaluation.fuel_evaluators import evaluate_ebitda

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
