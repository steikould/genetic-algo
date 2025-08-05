# Contains logic for generating synthetic data for testing and development.
import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass

from ....domain.fuel.pipeline import PipelineNode
from ....domain.fuel.specifications import FuelSpec

class Inputs:
    def __init__(self):
        self.n_locations = 3
        self.S_transmix = np.array([0.8, 0.1, 0.1])
        self.S_target = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.5, 0.4, 0.1]])
        self.diesel_price = 100
        self.transport_cost_matrix = np.array([10, 20, 30])
        self.storage_cost_per_unit = 5
        self.target_specs = self.S_target

class PipelineDataManager:
    """Handles data retrieval from BigQuery or synthetic data"""

    def __init__(self, use_bigquery: bool = False):
        self.use_bigquery = use_bigquery

    def get_pipeline_nodes(self) -> List[PipelineNode]:
        """Retrieve pipeline node configurations"""
        if self.use_bigquery:
            return self._fetch_nodes_from_bigquery()
        else:
            return self._generate_synthetic_nodes()

    def get_fuel_specs(self) -> Dict[str, FuelSpec]:
        """Retrieve fuel specifications"""
        if self.use_bigquery:
            return self._fetch_fuel_specs_from_bigquery()
        else:
            return self._generate_synthetic_fuel_specs()

    def _generate_synthetic_nodes(self) -> List[PipelineNode]:
        """Generate synthetic pipeline network data"""
        nodes = [
            # Upstream node with full capabilities
            PipelineNode(
                node_id="N001",
                location_name="Refinery_A",
                has_transmix=True,
                transmix_volume=50000,  # barrels
                can_blend=True,
                can_reinject=True,
                max_blend_rate=1000,  # barrels/hour
                storage_capacity=200000,
                current_inventory=150000,
                transport_cost=10.0,  # $/barrel
                storage_cost_per_unit=0.5  # $/barrel/day
            ),
            # Intermediate nodes with blending but no reinjection
            PipelineNode(
                node_id="N002",
                location_name="Terminal_B",
                has_transmix=True,
                transmix_volume=30000,
                can_blend=True,
                can_reinject=False,
                max_blend_rate=800,
                storage_capacity=100000,
                current_inventory=75000,
                transport_cost=20.0,
                storage_cost_per_unit=0.7
            ),
            PipelineNode(
                node_id="N003",
                location_name="Terminal_C",
                has_transmix=True,
                transmix_volume=25000,
                can_blend=True,
                can_reinject=False,
                max_blend_rate=600,
                storage_capacity=80000,
                current_inventory=60000,
                transport_cost=30.0,
                storage_cost_per_unit=0.8
            ),
        ]
        return nodes

    def _generate_synthetic_fuel_specs(self) -> Dict[str, FuelSpec]:
        """Generate synthetic fuel specification data"""
        return {
            "REGULAR": FuelSpec("REGULAR", 87.0, 9.0, 30.0, 0.74, 3.50),
            "PREMIUM": FuelSpec("PREMIUM", 93.0, 8.5, 20.0, 0.75, 3.80),
            "DIESEL": FuelSpec("DIESEL", 0.0, 0.0, 15.0, 0.85, 3.60),
            "TRANSMIX": FuelSpec("TRANSMIX", 85.0, 9.5, 50.0, 0.76, 2.80),
        }

    def _fetch_nodes_from_bigquery(self) -> List[PipelineNode]:
        """Stub for BigQuery node data retrieval"""
        # TODO: Implement BigQuery connection
        raise NotImplementedError("BigQuery connection not implemented")

    def _fetch_fuel_specs_from_bigquery(self) -> Dict[str, FuelSpec]:
        """Stub for BigQuery fuel specs data retrieval"""
        # TODO: Implement BigQuery connection
        raise NotImplementedError("BigQuery connection not implemented")
