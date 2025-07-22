"""
Enhanced Fuel Pipeline Blending Optimization using Hybrid GA + Local Optimization
Incorporates smart initialization, stagnation detection, and local refinement
"""

import random
import numpy as np
from deap import base, creator, tools, algorithms
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

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
    transport_cost: float  # Added for EBITDA calculation
    storage_cost_per_unit: float  # Added for EBITDA calculation

@dataclass
class FuelSpec:
    """Fuel specifications and quality parameters"""
    fuel_type: str
    octane: float
    rvp: float  # Reid Vapor Pressure
    sulfur: float
    density: float
    market_price: float  # Added for EBITDA calculation

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

class EnhancedBlendingOptimizer:
    """Hybrid GA + Local Optimization for fuel blending"""
    
    def __init__(self, nodes: List[PipelineNode], fuel_specs: Dict[str, FuelSpec], 
                 optimization_mode: str = 'hybrid'):
        """
        optimization_mode: 'swell' (maximize swell volume), 
                          'ebitda' (maximize profit),
                          'hybrid' (multi-objective)
        """
        self.nodes = nodes
        self.fuel_specs = fuel_specs
        self.blending_nodes = [n for n in nodes if n.can_blend]
        self.optimization_mode = optimization_mode
        
        # GA parameters
        self.population_size = 200
        self.generations = 100
        self.crossover_prob = 0.6
        self.mutation_prob = 0.2
        self.mutation_sigma = 0.1
        self.tournament_size = 3
        self.elite_size = 5
        
        # Stagnation detection
        self.max_stagnation = 20
        self.stagnation_counter = 0
        self.last_best_fitness = None
        
        # Individual representation: blend ratios that sum to 1
        self.individual_size = len(self.blending_nodes)
        
        # Calculate node properties for smart initialization
        self._calculate_node_properties()
    
    def _calculate_node_properties(self):
        """Calculate properties needed for smart initialization"""
        # Transmix properties (average across nodes)
        total_transmix = sum(n.transmix_volume for n in self.blending_nodes)
        self.transmix_props = np.array([n.transmix_volume / total_transmix 
                                       for n in self.blending_nodes])
        
        # Target properties based on node capabilities
        self.target_props = []
        for node in self.blending_nodes:
            # Nodes with reinjection capability get higher target
            if node.can_reinject:
                target = 0.4
            else:
                target = 0.3
            self.target_props.append(target)
        self.target_props = np.array(self.target_props)
    
    def smart_seed(self, epsilon=1e-6):
        """
        Create smart initial population using Dirichlet distribution
        """
        # Calculate inverse-delta weights: higher weight = better match
        delta_weights = 1.0 / (np.abs(self.target_props - self.transmix_props) + epsilon)
        
        # Generate population using Dirichlet distribution
        init_population = []
        for _ in range(self.population_size):
            alpha = delta_weights * 20  # Concentration parameter
            sample = np.random.dirichlet(alpha)
            init_population.append(sample.tolist())
        
        return init_population
    
    def setup_ga(self):
        """Configure DEAP genetic algorithm with enhanced features"""
        # Create fitness class based on optimization mode
        if self.optimization_mode == 'hybrid':
            # Multi-objective: maximize both swell and EBITDA
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
        else:
            # Single objective
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        creator.create("Individual", list, 
                      fitness=creator.FitnessMulti if self.optimization_mode == 'hybrid' 
                      else creator.FitnessMax)
        
        # Initialize toolbox
        self.toolbox = base.Toolbox()
        
        # Register operators
        self.toolbox.register("evaluate", self.evaluate_comprehensive)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_and_normalize)
        self.toolbox.register("select", tools.selTournament, 
                            tournsize=self.tournament_size)
    
    def mutate_and_normalize(self, individual, mu=0):
        """Apply Gaussian mutation then renormalize to sum=1"""
        # Apply Gaussian mutation
        for i in range(len(individual)):
            if random.random() < self.mutation_prob:
                individual[i] += random.gauss(mu, self.mutation_sigma)
        
        # Ensure non-negative and normalize
        individual[:] = np.clip(individual, 0, 1)
        total = sum(individual)
        if total > 0:
            individual[:] = [x / total for x in individual]
        else:
            # If all zeros, reset to uniform
            individual[:] = [1.0 / len(individual)] * len(individual)
        
        return individual,
    
    def calculate_swell_volume(self, allocation: np.ndarray) -> float:
        """Calculate total swell volume for given allocation"""
        total_swell = 0.0
        
        for i, (node, alloc) in enumerate(zip(self.blending_nodes, allocation)):
            if alloc > 0 and node.transmix_volume > 0:
                # Amount of transmix to blend
                transmix_blend = node.transmix_volume * alloc
                
                # Calculate swell based on density differences
                base_fuel_density = 0.74  # Average for gasoline
                transmix_density = self.fuel_specs["TRANSMIX"].density
                
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
    
    def calculate_ebitda(self, allocation: np.ndarray) -> float:
        """Calculate EBITDA for given allocation"""
        total_ebitda = 0.0
        
        for i, (node, alloc) in enumerate(zip(self.blending_nodes, allocation)):
            if alloc > 0:
                # Calculate revenue from blending
                transmix_qty = node.transmix_volume * alloc
                swell_volume = self.calculate_swell_volume(np.array([alloc]))
                
                # Revenue from selling blended product
                blend_price = self.fuel_specs["REGULAR"].market_price
                revenue = blend_price * (transmix_qty + swell_volume)
                
                # Calculate costs
                transport_cost = node.transport_cost * transmix_qty
                storage_cost = node.storage_cost_per_unit * transmix_qty
                blending_cost = 10.0 * transmix_qty  # Fixed blending cost
                
                # Location EBITDA
                location_ebitda = revenue - transport_cost - storage_cost - blending_cost
                total_ebitda += location_ebitda
        
        return total_ebitda
    
    def evaluate_comprehensive(self, individual: List[float]) -> Tuple[float, ...]:
        """Comprehensive evaluation function"""
        allocation = np.array(individual)
        
        # Calculate metrics
        swell_volume = self.calculate_swell_volume(allocation)
        ebitda = self.calculate_ebitda(allocation)
        
        # Apply constraints
        penalty = 0
        
        # Check blend rate constraints
        for i, (node, alloc) in enumerate(zip(self.blending_nodes, allocation)):
            transmix_blend = node.transmix_volume * alloc
            blend_time = transmix_blend / node.max_blend_rate if node.max_blend_rate > 0 else 0
            if blend_time > 24:  # Max 24 hours
                penalty += (blend_time - 24) * 1000
        
        # Apply penalties
        if penalty > 0:
            swell_volume *= 0.5
            ebitda -= penalty
        
        # Return based on optimization mode
        if self.optimization_mode == 'swell':
            return (swell_volume,)
        elif self.optimization_mode == 'ebitda':
            return (ebitda,)
        else:  # hybrid
            return (swell_volume, ebitda)
    
    def run_genetic_algorithm(self) -> Tuple[np.ndarray, float]:
        """Run enhanced genetic algorithm with stagnation detection"""
        self.setup_ga()
        
        # Smart initialization
        logger.info("Generating smart initial population...")
        init_allocations = self.smart_seed()
        population = [creator.Individual(alloc) for alloc in init_allocations]
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Statistics
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        if self.optimization_mode == 'hybrid':
            stats.register("avg", lambda x: np.mean(x, axis=0))
            stats.register("max", lambda x: np.max(x, axis=0))
        else:
            stats.register("avg", np.mean)
            stats.register("max", np.max)
        
        # Evolution loop with stagnation detection
        logger.info("Starting evolution...")
        logbook = tools.Logbook()
        
        for generation in range(self.generations):
            # Select elite individuals
            elite = tools.selBest(population, k=self.elite_size)
            
            # Create offspring
            offspring = algorithms.varAnd(population, self.toolbox, 
                                        self.crossover_prob, self.mutation_prob)
            
            # Evaluate offspring
            fits = list(map(self.toolbox.evaluate, offspring))
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            
            # Select next generation (elite + selection from offspring)
            selected = self.toolbox.select(offspring, k=len(population) - self.elite_size)
            population = elite + selected
            
            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=generation, **record)
            
            # Check for stagnation
            current_best = tools.selBest(population, k=1)[0]
            if self.optimization_mode == 'hybrid':
                current_best_value = sum(current_best.fitness.values)
            else:
                current_best_value = current_best.fitness.values[0]
            
            if self.last_best_fitness is None or current_best_value > self.last_best_fitness:
                self.last_best_fitness = current_best_value
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            # Early stopping
            if self.stagnation_counter >= self.max_stagnation:
                logger.info(f"Converged at generation {generation}")
                break
            
            # Log progress
            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {current_best_value:.2f}")
        
        # Get best solution
        best = tools.selBest(population, k=1)[0]
        best_allocation = np.array(best)
        
        # Ensure normalization
        best_allocation = np.clip(best_allocation, 0, 1)
        best_allocation = best_allocation / np.sum(best_allocation)
        
        return best_allocation, self.last_best_fitness, logbook
    
    def apply_local_optimization(self, initial_allocation: np.ndarray, 
                               method: str = 'SLSQP') -> np.ndarray:
        """Apply local optimization to refine GA solution"""
        logger.info(f"Applying local optimization with {method}...")
        
        # Constraint: allocations must sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        
        # Bounds: each allocation between 0 and 1
        bounds = [(0, 1) for _ in range(len(initial_allocation))]
        
        # Objective function (negate for minimization)
        def objective(x):
            if self.optimization_mode == 'swell':
                return -self.calculate_swell_volume(x)
            elif self.optimization_mode == 'ebitda':
                return -self.calculate_ebitda(x)
            else:  # hybrid - weighted sum
                swell = self.calculate_swell_volume(x)
                ebitda = self.calculate_ebitda(x)
                # Normalize and weight equally
                return -(swell / 1000 + ebitda / 10000)
        
        # Run optimization
        result = minimize(
            objective,
            initial_allocation,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if result.success:
            logger.info(f"Local optimization converged: {result.message}")
            refined_allocation = result.x
            # Ensure normalization
            refined_allocation = np.clip(refined_allocation, 0, 1)
            refined_allocation = refined_allocation / np.sum(refined_allocation)
            return refined_allocation
        else:
            logger.warning(f"Local optimization failed: {result.message}")
            return initial_allocation
    
    def optimize(self, use_local_refinement: bool = True) -> Dict:
        """Run complete optimization process"""
        # Run genetic algorithm
        ga_allocation, ga_fitness, logbook = self.run_genetic_algorithm()
        
        # Calculate GA metrics
        ga_swell = self.calculate_swell_volume(ga_allocation)
        ga_ebitda = self.calculate_ebitda(ga_allocation)
        
        results = {
            'ga_allocation': ga_allocation,
            'ga_fitness': ga_fitness,
            'ga_swell_volume': ga_swell,
            'ga_ebitda': ga_ebitda,
            'convergence_history': logbook
        }
        
        # Apply local refinement if requested
        if use_local_refinement:
            refined_allocation = self.apply_local_optimization(ga_allocation)
            refined_swell = self.calculate_swell_volume(refined_allocation)
            refined_ebitda = self.calculate_ebitda(refined_allocation)
            
            results.update({
                'refined_allocation': refined_allocation,
                'refined_swell_volume': refined_swell,
                'refined_ebitda': refined_ebitda,
                'improvement_swell': ((refined_swell - ga_swell) / ga_swell * 100) if ga_swell > 0 else 0,
                'improvement_ebitda': ((refined_ebitda - ga_ebitda) / ga_ebitda * 100) if ga_ebitda > 0 else 0
            })
        
        # Create blend plan
        blend_plan = {}
        final_allocation = results.get('refined_allocation', ga_allocation)
        
        for i, (node, alloc) in enumerate(zip(self.blending_nodes, final_allocation)):
            blend_plan[node.node_id] = {
                'location': node.location_name,
                'allocation_ratio': alloc,
                'transmix_volume': node.transmix_volume,
                'blend_volume': node.transmix_volume * alloc,
                'can_reinject': node.can_reinject,
                'transport_cost': node.transport_cost,
                'estimated_swell': self.calculate_swell_volume(np.array([alloc])),
                'estimated_profit': self.calculate_ebitda(np.array([alloc]))
            }
        
        results['blend_plan'] = blend_plan
        return results

def visualize_optimization_results(results: Dict, optimizer: EnhancedBlendingOptimizer):
    """Create comprehensive visualizations of optimization results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Allocation comparison (GA vs Refined)
    ax1 = axes[0, 0]
    locations = [n.location_name for n in optimizer.blending_nodes]
    
    if 'refined_allocation' in results:
        x = np.arange(len(locations))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, results['ga_allocation'], width, 
                        label='GA Solution', alpha=0.8)
        bars2 = ax1.bar(x + width/2, results['refined_allocation'], width, 
                        label='Refined Solution', alpha=0.8)
        
        ax1.set_ylabel('Allocation Ratio')
        ax1.set_title('Allocation Comparison: GA vs Local Refinement')
        ax1.set_xticks(x)
        ax1.set_xticklabels(locations, rotation=45, ha='right')
        ax1.legend()
    else:
        bars = ax1.bar(locations, results['ga_allocation'])
        ax1.set_ylabel('Allocation Ratio')
        ax1.set_title('Optimal Allocation from GA')
        ax1.set_xticklabels(locations, rotation=45, ha='right')
    
    # 2. Volume breakdown
    ax2 = axes[0, 1]
    transmix_volumes = []
    blend_volumes = []
    
    for node_id, plan in results['blend_plan'].items():
        transmix_volumes.append(plan['transmix_volume'])
        blend_volumes.append(plan['blend_volume'])
    
    x = np.arange(len(locations))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, transmix_volumes, width, 
                    label='Available Transmix', alpha=0.8)
    bars2 = ax2.bar(x + width/2, blend_volumes, width, 
                    label='Allocated Volume', alpha=0.8)
    
    ax2.set_title('Transmix Allocation by Location')
    ax2.set_ylabel('Volume (barrels)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(locations, rotation=45, ha='right')
    ax2.legend()
    
    # 3. Convergence history
    ax3 = axes[0, 2]
    gen = results['convergence_history'].select("gen")
    
    if optimizer.optimization_mode == 'hybrid':
        # Plot both objectives
        max_vals = np.array(results['convergence_history'].select("max"))
        ax3.plot(gen, max_vals[:, 0], 'b-', label='Swell Volume', linewidth=2)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(gen, max_vals[:, 1], 'r-', label='EBITDA', linewidth=2)
        ax3_twin.set_ylabel('EBITDA', color='r')
        ax3_twin.tick_params(axis='y', labelcolor='r')
    else:
        fit_max = results['convergence_history'].select("max")
        fit_avg = results['convergence_history'].select("avg")
        ax3.plot(gen, fit_max, 'b-', label='Best Fitness', linewidth=2)
        ax3.plot(gen, fit_avg, 'r--', label='Average Fitness', linewidth=2)
    
    ax3.set_title('GA Convergence History')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Fitness')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Financial metrics by location
    ax4 = axes[1, 0]
    profits = [plan['estimated_profit'] for plan in results['blend_plan'].values()]
    colors = ['green' if p > 0 else 'red' for p in profits]
    
    bars = ax4.bar(locations, profits, color=colors)
    ax4.set_title('Estimated Profit by Location')
    ax4.set_ylabel('EBITDA ($)')
    ax4.set_xticklabels(locations, rotation=45, ha='right')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 5. Swell volume by location
    ax5 = axes[1, 1]
    swells = [plan['estimated_swell'] for plan in results['blend_plan'].values()]
    
    bars = ax5.bar(locations, swells, color='skyblue')
    ax5.set_title('Estimated Swell Volume by Location')
    ax5.set_ylabel('Swell Volume (barrels)')
    ax5.set_xticklabels(locations, rotation=45, ha='right')
    
    # 6. Summary metrics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    if 'refined_allocation' in results:
        summary_text = f"""
        OPTIMIZATION SUMMARY
        
        GA Results:
        - Swell Volume: {results['ga_swell_volume']:,.0f} barrels
        - EBITDA: ${results['ga_ebitda']:,.2f}
        
        Refined Results:
        - Swell Volume: {results['refined_swell_volume']:,.0f} barrels
        - EBITDA: ${results['refined_ebitda']:,.2f}
        
        Improvements:
        - Swell: {results['improvement_swell']:.1f}%
        - EBITDA: {results['improvement_ebitda']:.1f}%
        
        Total Transmix: {sum(n.transmix_volume for n in optimizer.blending_nodes):,.0f} barrels
        Total Allocated: {sum(plan['blend_volume'] for plan in results['blend_plan'].values()):,.0f} barrels
        """
    else:
        summary_text = f"""
        OPTIMIZATION SUMMARY
        
        Results:
        - Swell Volume: {results['ga_swell_volume']:,.0f} barrels
        - EBITDA: ${results['ga_ebitda']:,.2f}
        
        Total Transmix: {sum(n.transmix_volume for n in optimizer.blending_nodes):,.0f} barrels
        Total Allocated: {sum(plan['blend_volume'] for plan in results['blend_plan'].values()):,.0f} barrels
        """
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, 
             fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('enhanced_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_optimization_modes():
    """Compare different optimization modes"""
    logger.info("Comparing optimization modes...")
    
    # Initialize data
    manager = PipelineDataManager(use_bigquery=False)
    nodes = manager.get_pipeline_nodes()
    fuel_specs = manager.get_fuel_specs()
    
    modes = ['swell', 'ebitda', 'hybrid']
    results_comparison = {}
    
    for mode in modes:
        logger.info(f"\nOptimizing for mode: {mode}")
        optimizer = EnhancedBlendingOptimizer(nodes, fuel_specs, optimization_mode=mode)
        optimizer.population_size = 100
        optimizer.generations = 50
        
        results = optimizer.optimize(use_local_refinement=True)
        results_comparison[mode] = results
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (mode, results) in enumerate(results_comparison.items()):
        ax = axes[idx]
        locations = [n.location_name for n in optimizer.blending_nodes]
        allocation = results.get('refined_allocation', results['ga_allocation'])
        
        bars = ax.bar(locations, allocation)
        ax.set_title(f'{mode.upper()} Optimization')
        ax.set_ylabel('Allocation Ratio')
        ax.set_ylim(0, 1)
        ax.set_xticklabels(locations, rotation=45, ha='right')
        
        # Add metrics text
        swell = results.get('refined_swell_volume', results['ga_swell_volume'])
        ebitda = results.get('refined_ebitda', results['ga_ebitda'])
        ax.text(0.5, 0.95, f'Swell: {swell:.0f}\nEBITDA: ${ebitda:.0f}', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('optimization_mode_comparison.png', dpi=300)
    plt.show()
    
    return results_comparison

def main():
    """Main execution function with enhanced optimizer"""
    logger.info("Initializing enhanced fuel pipeline blending optimization...")
    
    # Initialize data manager
    manager = PipelineDataManager(use_bigquery=USE_BIGQUERY)
    
    # Load data
    logger.info("Loading pipeline network data...")
    nodes = manager.get_pipeline_nodes()
    fuel_specs = manager.get_fuel_specs()
    
    # Initialize enhanced optimizer (hybrid mode)
    optimizer = EnhancedBlendingOptimizer(nodes, fuel_specs, optimization_mode='hybrid')
    
    # Run optimization with local refinement
    logger.info("Running hybrid GA + local optimization...")
    results = optimizer.optimize(use_local_refinement=True)
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*60)
    
    logger.info("\nGA Solution:")
    logger.info(f"  Swell Volume: {results['ga_swell_volume']:,.0f} barrels")
    logger.info(f"  EBITDA: ${results['ga_ebitda']:,.2f}")
    
    if 'refined_allocation' in results:
        logger.info("\nRefined Solution (after local optimization):")
        logger.info(f"  Swell Volume: {results['refined_swell_volume']:,.0f} barrels")
        logger.info(f"  EBITDA: ${results['refined_ebitda']:,.2f}")
        logger.info(f"  Swell Improvement: {results['improvement_swell']:.1f}%")
        logger.info(f"  EBITDA Improvement: {results['improvement_ebitda']:.1f}%")
    
    logger.info("\nOptimal Allocation Plan:")
    for node_id, plan in results['blend_plan'].items():
        logger.info(f"\n{plan['location']} ({node_id}):")
        logger.info(f"  - Allocation Ratio: {plan['allocation_ratio']:.2%}")
        logger.info(f"  - Transmix Available: {plan['transmix_volume']:,.0f} barrels")
        logger.info(f"  - Allocated Volume: {plan['blend_volume']:,.0f} barrels")
        logger.info(f"  - Estimated Profit: ${plan['estimated_profit']:,.2f}")
    
    # Create visualizations
    visualize_optimization_results(results, optimizer)
    
    # Compare optimization modes
    comparison_results = compare_optimization_modes()
    
    # Export results
    results_df = pd.DataFrame.from_dict(results['blend_plan'], orient='index')
    results_df.to_csv('enhanced_blend_optimization_results.csv')
    logger.info("\nResults saved to 'enhanced_blend_optimization_results.csv'")
    
    return results, comparison_results

if __name__ == "__main__":
    results, comparison = main()