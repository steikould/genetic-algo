"""
Fuel Pipeline Blending Optimization using Genetic Algorithm
Optimizes swell volume when blending fuels in a pipeline network with constraints
"""

import random
import numpy as np
from deap import base, creator, tools, algorithms
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
USE_BIGQUERY = False  # Toggle for database vs synthetic data

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

@dataclass
class FuelSpec:
    """Fuel specifications and quality parameters"""
    fuel_type: str
    octane: float
    rvp: float  # Reid Vapor Pressure
    sulfur: float
    density: float

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
    
    def get_pipeline_batches(self) -> pd.DataFrame:
        """Retrieve current pipeline batch schedule"""
        if self.use_bigquery:
            return self._fetch_batches_from_bigquery()
        else:
            return self._generate_synthetic_batches()
    
    def _fetch_nodes_from_bigquery(self) -> List[PipelineNode]:
        """Stub for BigQuery node data retrieval"""
        # TODO: Implement BigQuery connection
        # query = """
        # SELECT node_id, location_name, has_transmix, transmix_volume,
        #        can_blend, can_reinject, max_blend_rate, storage_capacity,
        #        current_inventory
        # FROM `project.dataset.pipeline_nodes`
        # WHERE active = TRUE
        # """
        raise NotImplementedError("BigQuery connection not implemented")
    
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
                current_inventory=150000
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
                current_inventory=75000
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
                current_inventory=60000
            ),
            # Downstream nodes without blending capability
            PipelineNode(
                node_id="N004",
                location_name="Terminal_D",
                has_transmix=False,
                transmix_volume=0,
                can_blend=False,
                can_reinject=False,
                max_blend_rate=0,
                storage_capacity=50000,
                current_inventory=40000
            ),
            PipelineNode(
                node_id="N005",
                location_name="Terminal_E",
                has_transmix=False,
                transmix_volume=0,
                can_blend=False,
                can_reinject=False,
                max_blend_rate=0,
                storage_capacity=60000,
                current_inventory=45000
            ),
        ]
        return nodes
    
    def _generate_synthetic_fuel_specs(self) -> Dict[str, FuelSpec]:
        """Generate synthetic fuel specification data"""
        return {
            "REGULAR": FuelSpec("REGULAR", 87.0, 9.0, 30.0, 0.74),
            "PREMIUM": FuelSpec("PREMIUM", 93.0, 8.5, 20.0, 0.75),
            "DIESEL": FuelSpec("DIESEL", 0.0, 0.0, 15.0, 0.85),
            "TRANSMIX": FuelSpec("TRANSMIX", 85.0, 9.5, 50.0, 0.76),
        }
    
    def _generate_synthetic_batches(self) -> pd.DataFrame:
        """Generate synthetic batch schedule data"""
        batches = pd.DataFrame({
            'batch_id': [f'B{i:03d}' for i in range(1, 11)],
            'product': ['REGULAR', 'PREMIUM', 'DIESEL'] * 3 + ['REGULAR'],
            'volume': np.random.uniform(10000, 50000, 10),
            'destination_node': np.random.choice(['N004', 'N005'], 10),
            'scheduled_time': pd.date_range(start='2024-01-01', periods=10, freq='6H')
        })
        return batches

class BlendingOptimizer:
    """Genetic Algorithm optimizer for fuel blending"""
    
    def __init__(self, nodes: List[PipelineNode], fuel_specs: Dict[str, FuelSpec], 
                 batches: pd.DataFrame):
        self.nodes = nodes
        self.fuel_specs = fuel_specs
        self.batches = batches
        self.blending_nodes = [n for n in nodes if n.can_blend]
        
        # GA parameters
        self.population_size = 200
        self.generations = 100
        self.crossover_prob = 0.8
        self.mutation_prob = 0.2
        self.tournament_size = 3
        
        # Individual representation: blend ratios for each blending node
        self.individual_size = len(self.blending_nodes)
        
    def setup_ga(self):
        """Configure DEAP genetic algorithm"""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        
        # Attribute generator: blend ratio between 0 and 1
        self.toolbox.register("attr_float", random.uniform, 0.0, 1.0)
        
        # Individual and population generators
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_float, n=self.individual_size)
        self.toolbox.register("population", tools.initRepeat, list, 
                            self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_blend_strategy)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, 
                            tournsize=self.tournament_size)
        
    def evaluate_blend_strategy(self, individual: List[float]) -> Tuple[float,]:
        """
        Evaluate a blending strategy
        Returns tuple of (swell_volume,)
        """
        total_swell = 0.0
        constraints_violated = False
        
        # Map blend ratios to nodes
        blend_plan = {}
        for i, node in enumerate(self.blending_nodes):
            blend_ratio = max(0.0, min(1.0, individual[i]))  # Ensure 0-1 range
            blend_plan[node.node_id] = blend_ratio
        
        # Calculate swell for each blending operation
        for node in self.blending_nodes:
            if node.transmix_volume > 0:
                blend_ratio = blend_plan[node.node_id]
                
                # Amount of transmix to blend
                transmix_blend = node.transmix_volume * blend_ratio
                
                # Check blend rate constraint
                blend_time = transmix_blend / node.max_blend_rate
                if blend_time > 24:  # Max 24 hours for blending operation
                    constraints_violated = True
                
                # Calculate swell based on transmix properties
                # Swell occurs due to density differences and molecular interactions
                base_fuel_density = 0.74  # Average for gasoline
                transmix_density = self.fuel_specs["TRANSMIX"].density
                
                # Simplified swell calculation
                density_diff = abs(base_fuel_density - transmix_density)
                swell_factor = 1.0 + (density_diff * 0.1 * blend_ratio)
                
                # Volume swell
                volume_swell = transmix_blend * (swell_factor - 1.0)
                
                # Apply node-specific factors
                if node.can_reinject:
                    # Reinjection allows better control and higher swell
                    volume_swell *= 1.2
                
                total_swell += volume_swell
        
        # Apply penalties for constraint violations
        if constraints_violated:
            total_swell *= 0.5
        
        # Check if total blended volume exceeds pipeline capacity
        total_blended = sum(node.transmix_volume * blend_plan.get(node.node_id, 0) 
                          for node in self.blending_nodes)
        if total_blended > 100000:  # Pipeline capacity limit
            total_swell *= 0.7
        
        return (total_swell,)
    
    def optimize(self) -> Dict:
        """Run genetic algorithm optimization"""
        self.setup_ga()
        
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        
        # Statistics
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of fame to track best individuals
        hof = tools.HallOfFame(1)
        
        # Run GA
        logger.info("Starting genetic algorithm optimization...")
        population, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )
        
        # Extract best solution
        best_individual = hof[0]
        best_fitness = best_individual.fitness.values[0]
        
        # Create blend plan from best individual
        blend_plan = {}
        for i, node in enumerate(self.blending_nodes):
            blend_ratio = max(0.0, min(1.0, best_individual[i]))
            blend_plan[node.node_id] = {
                'location': node.location_name,
                'blend_ratio': blend_ratio,
                'transmix_volume': node.transmix_volume,
                'blend_volume': node.transmix_volume * blend_ratio,
                'can_reinject': node.can_reinject
            }
        
        return {
            'best_fitness': best_fitness,
            'total_swell_volume': best_fitness,
            'blend_plan': blend_plan,
            'convergence_history': logbook
        }

def main():
    """Main execution function"""
    logger.info("Initializing fuel pipeline blending optimization...")
    
    # Initialize data manager
    data_manager = PipelineDataManager(use_bigquery=USE_BIGQUERY)
    
    # Load data
    logger.info("Loading pipeline network data...")
    nodes = data_manager.get_pipeline_nodes()
    fuel_specs = data_manager.get_fuel_specs()
    batches = data_manager.get_pipeline_batches()
    
    logger.info(f"Loaded {len(nodes)} pipeline nodes")
    logger.info(f"Blending nodes: {sum(1 for n in nodes if n.can_blend)}")
    
    # Initialize optimizer
    optimizer = BlendingOptimizer(nodes, fuel_specs, batches)
    
    # Run optimization
    results = optimizer.optimize()
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*50)
    logger.info(f"Maximum Swell Volume: {results['best_fitness']:.2f} barrels")
    logger.info("\nOptimal Blend Plan:")
    
    for node_id, plan in results['blend_plan'].items():
        logger.info(f"\n{plan['location']} ({node_id}):")
        logger.info(f"  - Blend Ratio: {plan['blend_ratio']:.2%}")
        logger.info(f"  - Transmix Available: {plan['transmix_volume']:,.0f} barrels")
        logger.info(f"  - Blend Volume: {plan['blend_volume']:,.0f} barrels")
        logger.info(f"  - Can Reinject: {plan['can_reinject']}")
    
    # Export results
    results_df = pd.DataFrame.from_dict(results['blend_plan'], orient='index')
    results_df.to_csv('blend_optimization_results.csv')
    logger.info("\nResults saved to 'blend_optimization_results.csv'")
    
    return results

if __name__ == "__main__":
    results = main()