# Contains the domain logic related to pipelines.
from dataclasses import dataclass

@dataclass
class PipelineNode:
    """Represents a node in the pipeline network"""
    node_id: str
    location_name: str
    has_transmix: bool
    transmix_volume: float
    can_blend: bool
    can_reinject: bool
    max_blend_rate: float
    storage_capacity: float
    current_inventory: float
    transport_cost: float  # Added for EBITDA calculation
    storage_cost_per_unit: float  # Added for EBITDA calculation
