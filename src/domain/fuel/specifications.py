# Defines the specifications and properties of fuels.
from dataclasses import dataclass

@dataclass
class FuelSpec:
    """Fuel specifications and quality parameters"""
    fuel_type: str
    octane: float
    rvp: float  # Reid Vapor Pressure
    sulfur: float
    density: float
    market_price: float  # Added for EBITDA calculation
