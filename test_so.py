"""
Test suite for Fuel Pipeline Blending Optimization
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import logging

# Import the main module (assuming it's saved as 'pipeline_optimizer.py')
from single_objective import (
    PipelineNode, FuelSpec, PipelineDataManager, 
    BlendingOptimizer, main
)

class TestPipelineOptimizer(unittest.TestCase):
    """Unit tests for pipeline optimization components"""
    
    def setUp(self):
        """Set up test data"""
        self.test_nodes = [
            PipelineNode("N001", "Test_Refinery", True, 50000, True, True, 1000, 200000, 150000),
            PipelineNode("N002", "Test_Terminal", True, 30000, True, False, 800, 100000, 75000),
            PipelineNode("N003", "Test_Storage", False, 0, False, False, 0, 50000, 40000),
        ]
        
        self.test_fuel_specs = {
            "REGULAR": FuelSpec("REGULAR", 87.0, 9.0, 30.0, 0.74),
            "TRANSMIX": FuelSpec("TRANSMIX", 85.0, 9.5, 50.0, 0.76),
        }
        
        self.test_batches = pd.DataFrame({
            'batch_id': ['B001', 'B002'],
            'product': ['REGULAR', 'PREMIUM'],
            'volume': [20000, 30000],
            'destination_node': ['N003', 'N003'],
            'scheduled_time': pd.date_range(start='2024-01-01', periods=2, freq='12H')
        })
    
    def test_pipeline_node_creation(self):
        """Test PipelineNode dataclass"""
        node = self.test_nodes[0]
        self.assertEqual(node.node_id, "N001")
        self.assertTrue(node.can_reinject)
        self.assertEqual(node.transmix_volume, 50000)
    
    def test_data_manager_synthetic(self):
        """Test synthetic data generation"""
        manager = PipelineDataManager(use_bigquery=False)
        nodes = manager.get_pipeline_nodes()
        fuel_specs = manager.get_fuel_specs()
        batches = manager.get_pipeline_batches()
        
        self.assertGreater(len(nodes), 0)
        self.assertIn("REGULAR", fuel_specs)
        self.assertIsInstance(batches, pd.DataFrame)
    
    def test_optimizer_initialization(self):
        """Test BlendingOptimizer initialization"""
        optimizer = BlendingOptimizer(self.test_nodes, self.test_fuel_specs, self.test_batches)
        self.assertEqual(len(optimizer.blending_nodes), 2)  # Only 2 nodes can blend
        self.assertEqual(optimizer.individual_size, 2)
    
    def test_fitness_evaluation(self):
        """Test fitness function evaluation"""
        optimizer = BlendingOptimizer(self.test_nodes, self.test_fuel_specs, self.test_batches)
        optimizer.setup_ga()
        
        # Test individual with 50% blend ratio for each node
        test_individual = [0.5, 0.5]
        fitness = optimizer.evaluate_blend_strategy(test_individual)
        
        self.assertIsInstance(fitness, tuple)
        self.assertEqual(len(fitness), 1)
        self.assertGreater(fitness[0], 0)  # Should have positive swell
    
    def test_constraint_violations(self):
        """Test that constraints are properly enforced"""
        optimizer = BlendingOptimizer(self.test_nodes, self.test_fuel_specs, self.test_batches)
        optimizer.setup_ga()
        
        # Test with extreme blend ratios
        extreme_individual = [1.0, 1.0]  # 100% blend for all nodes
        fitness_extreme = optimizer.evaluate_blend_strategy(extreme_individual)
        
        # Test with moderate blend ratios
        moderate_individual = [0.3, 0.3]
        fitness_moderate = optimizer.evaluate_blend_strategy(moderate_individual)
        
        # Moderate should perform better due to constraints
        self.assertLess(fitness_extreme[0], fitness_moderate[0] * 1.5)

class IntegrationTest(unittest.TestCase):
    """Integration tests for the complete optimization process"""
    
    def test_full_optimization_run(self):
        """Test a complete optimization run with reduced parameters"""
        # Create test data
        manager = PipelineDataManager(use_bigquery=False)
        nodes = manager.get_pipeline_nodes()
        fuel_specs = manager.get_fuel_specs()
        batches = manager.get_pipeline_batches()
        
        # Initialize optimizer with smaller parameters for testing
        optimizer = BlendingOptimizer(nodes, fuel_specs, batches)
        optimizer.population_size = 20
        optimizer.generations = 10
        
        # Run optimization
        results = optimizer.optimize()
        
        # Verify results structure
        self.assertIn('best_fitness', results)
        self.assertIn('blend_plan', results)
        self.assertIn('convergence_history', results)
        
        # Verify blend plan
        for node_id, plan in results['blend_plan'].items():
            self.assertIn('blend_ratio', plan)
            self.assertGreaterEqual(plan['blend_ratio'], 0)
            self.assertLessEqual(plan['blend_ratio'], 1)

def run_example_optimization():
    """Run an example optimization and visualize results"""
    print("\n" + "="*60)
    print("RUNNING EXAMPLE OPTIMIZATION")
    print("="*60)
    
    # Initialize components
    manager = PipelineDataManager(use_bigquery=False)
    nodes = manager.get_pipeline_nodes()
    fuel_specs = manager.get_fuel_specs()
    batches = manager.get_pipeline_batches()
    
    # Create optimizer with moderate parameters
    optimizer = BlendingOptimizer(nodes, fuel_specs, batches)
    optimizer.population_size = 50
    optimizer.generations = 30
    
    # Run optimization
    results = optimizer.optimize()
    
    # Create visualizations
    create_visualizations(results, nodes)
    
    return results

def create_visualizations(results: Dict, nodes: List[PipelineNode]):
    """Create visualizations of optimization results"""
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Blend ratios by location
    ax1 = axes[0, 0]
    locations = []
    blend_ratios = []
    colors = []
    
    for node_id, plan in results['blend_plan'].items():
        locations.append(plan['location'])
        blend_ratios.append(plan['blend_ratio'])
        colors.append('green' if plan['can_reinject'] else 'orange')
    
    bars = ax1.bar(locations, blend_ratios, color=colors)
    ax1.set_title('Optimal Blend Ratios by Location', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Blend Ratio')
    ax1.set_ylim(0, 1.1)
    ax1.set_xticklabels(locations, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, ratio in zip(bars, blend_ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{ratio:.1%}', ha='center', va='bottom')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Can Reinject'),
        Patch(facecolor='orange', label='Cannot Reinject')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # 2. Volume breakdown
    ax2 = axes[0, 1]
    locations = []
    transmix_volumes = []
    blend_volumes = []
    
    for node_id, plan in results['blend_plan'].items():
        locations.append(plan['location'])
        transmix_volumes.append(plan['transmix_volume'])
        blend_volumes.append(plan['blend_volume'])
    
    x = np.arange(len(locations))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, transmix_volumes, width, label='Available Transmix', alpha=0.8)
    bars2 = ax2.bar(x + width/2, blend_volumes, width, label='Blended Volume', alpha=0.8)
    
    ax2.set_title('Transmix Volumes by Location', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Volume (barrels)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(locations, rotation=45, ha='right')
    ax2.legend()
    
    # 3. Convergence history
    ax3 = axes[1, 0]
    gen = results['convergence_history'].select("gen")
    fit_max = results['convergence_history'].select("max")
    fit_avg = results['convergence_history'].select("avg")
    
    ax3.plot(gen, fit_max, 'b-', label='Best Fitness', linewidth=2)
    ax3.plot(gen, fit_avg, 'r--', label='Average Fitness', linewidth=2)
    ax3.fill_between(gen, fit_avg, fit_max, alpha=0.3)
    
    ax3.set_title('GA Convergence History', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Fitness (Swell Volume)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    OPTIMIZATION SUMMARY
    
    Total Swell Volume: {results['best_fitness']:,.0f} barrels
    
    Number of Blending Locations: {len(results['blend_plan'])}
    
    Total Transmix Available: {sum(plan['transmix_volume'] for plan in results['blend_plan'].values()):,.0f} barrels
    
    Total Transmix Blended: {sum(plan['blend_volume'] for plan in results['blend_plan'].values()):,.0f} barrels
    
    Average Blend Ratio: {np.mean([plan['blend_ratio'] for plan in results['blend_plan'].values()]):.1%}
    
    Locations with Reinjection: {sum(1 for plan in results['blend_plan'].values() if plan['can_reinject'])}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved as 'optimization_results.png'")

def run_sensitivity_analysis():
    """Run sensitivity analysis on key parameters"""
    print("\n" + "="*60)
    print("RUNNING SENSITIVITY ANALYSIS")
    print("="*60)
    
    manager = PipelineDataManager(use_bigquery=False)
    nodes = manager.get_pipeline_nodes()
    fuel_specs = manager.get_fuel_specs()
    batches = manager.get_pipeline_batches()
    
    # Parameters to test
    population_sizes = [20, 50, 100]
    mutation_rates = [0.1, 0.2, 0.3]
    
    results_matrix = []
    
    for pop_size in population_sizes:
        for mut_rate in mutation_rates:
            print(f"\nTesting: Population={pop_size}, Mutation Rate={mut_rate}")
            
            optimizer = BlendingOptimizer(nodes, fuel_specs, batches)
            optimizer.population_size = pop_size
            optimizer.generations = 20
            optimizer.mutation_prob = mut_rate
            
            results = optimizer.optimize()
            
            results_matrix.append({
                'population_size': pop_size,
                'mutation_rate': mut_rate,
                'best_fitness': results['best_fitness'],
                'convergence_gen': len(results['convergence_history']) - 1
            })
    
    # Create sensitivity analysis visualization
    df_results = pd.DataFrame(results_matrix)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heatmap of fitness values
    pivot_fitness = df_results.pivot(index='mutation_rate', 
                                    columns='population_size', 
                                    values='best_fitness')
    sns.heatmap(pivot_fitness, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1)
    ax1.set_title('Best Fitness by Parameters')
    
    # Line plot of convergence
    for pop_size in population_sizes:
        data = df_results[df_results['population_size'] == pop_size]
        ax2.plot(data['mutation_rate'], data['best_fitness'], 
                marker='o', label=f'Pop Size: {pop_size}')
    
    ax2.set_xlabel('Mutation Rate')
    ax2.set_ylabel('Best Fitness')
    ax2.set_title('Parameter Sensitivity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300)
    plt.show()
    
    print("\nSensitivity analysis saved as 'sensitivity_analysis.png'")

if __name__ == "__main__":
    # Run tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run example optimization
    example_results = run_example_optimization()
    
    # Run sensitivity analysis
    run_sensitivity_analysis()
    
    # Run the main optimization
    print("\n" + "="*60)
    print("RUNNING MAIN OPTIMIZATION")
    print("="*60)
    main_results = main()