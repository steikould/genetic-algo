# Contains evaluators and fitness functions specific to the fuel blending domain.
import numpy as np
from ..domain.fuel.pipeline import PipelineNode

def calculate_swell_volume(allocation: np.ndarray, blending_nodes: list, fuel_specs: dict) -> float:
    """Calculate total swell volume for given allocation"""
    total_swell = 0.0

    for i, (node, alloc) in enumerate(zip(blending_nodes, allocation)):
        if alloc > 0 and node.transmix_volume > 0:
            # Amount of transmix to blend
            transmix_blend = node.transmix_volume * alloc

            # Calculate swell based on density differences
            base_fuel_density = 0.74  # Average for gasoline
            transmix_density = fuel_specs["TRANSMIX"].density

            # Enhanced swell calculation
            density_diff = abs(base_fuel_density - transmix_density)
            swell_factor = 1.0 + (density_diff * 0.1 * alloc)

            # Apply node-specific factors
            if node.can_reinject:
                swell_factor *= 1.2  # Reinjection bonus

            # Volume swell
            volume_swell = transmix_blend * (swell_factor - 1.0)
            total_swell += volume_swell

    return total_swell

def calculate_ebitda(allocation: np.ndarray, blending_nodes: list, fuel_specs: dict) -> float:
    """Calculate EBITDA for given allocation"""
    total_ebitda = 0.0

    for i, (node, alloc) in enumerate(zip(blending_nodes, allocation)):
        if alloc > 0:
            # Calculate revenue from blending
            transmix_qty = node.transmix_volume * alloc
            swell_volume = calculate_swell_volume(np.array([alloc]), [node], fuel_specs)

            # Revenue from selling blended product
            blend_price = fuel_specs["REGULAR"].market_price
            revenue = blend_price * (transmix_qty + swell_volume)

            # Calculate costs
            transport_cost = node.transport_cost * transmix_qty
            storage_cost = node.storage_cost_per_unit * transmix_qty
            blending_cost = 10.0 * transmix_qty  # Fixed blending cost

            # Location EBITDA
            location_ebitda = revenue - transport_cost - storage_cost - blending_cost
            total_ebitda += location_ebitda

    return total_ebitda

def evaluate_comprehensive(individual: list, blending_nodes: list, fuel_specs: dict, optimization_mode: str) -> tuple:
    """Comprehensive evaluation function"""
    allocation = np.array(individual)

    # Calculate metrics
    swell_volume = calculate_swell_volume(allocation, blending_nodes, fuel_specs)
    ebitda = calculate_ebitda(allocation, blending_nodes, fuel_specs)

    # Apply constraints
    penalty = 0

    # Check blend rate constraints
    for i, (node, alloc) in enumerate(zip(blending_nodes, allocation)):
        transmix_blend = node.transmix_volume * alloc
        blend_time = transmix_blend / node.max_blend_rate if node.max_blend_rate > 0 else 0
        if blend_time > 24:  # Max 24 hours
            penalty += (blend_time - 24) * 1000

    # Apply penalties
    if penalty > 0:
        swell_volume *= 0.5
        ebitda -= penalty

    # Return based on optimization mode
    if optimization_mode == 'swell':
        return (swell_volume,)
    elif optimization_mode == 'ebitda':
        return (ebitda,)
    else:  # hybrid
        return (swell_volume, ebitda)
