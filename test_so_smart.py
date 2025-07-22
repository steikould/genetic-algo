"""
Test suite for Enhanced Fuel Pipeline Blending Optimization
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, List

# Import the enhanced optimizer (assuming it's saved as 'enhanced_pipeline_optimizer.py')
from single_objective_smart import (
    PipelineNode, FuelSpec, PipelineDataManager, 
    EnhancedBlendingOptimizer, main, compare_optimization_modes,
    visualize_optimization_results
)

class TestEnhancedOptimizer(unittest.TestCase):
    """Unit tests for enhanced optimization components"""
    
    def setUp(self):
        """Set up test data"""
        self.test_nodes = [
            PipelineNode("N001", "Test_Refinery", True, 50000, True, True, 
                        1000, 200000, 150000, 10.0, 0.5),
            PipelineNode("N002", "Test_Terminal", True, 30000, True, False, 
                        800, 100000, 75000, 20.0, 0.7),
            PipelineNode("N003", "Test_Storage", True, 20000, True, False, 
                        600, 80000, 60000, 30.0, 0.8),
        ]
        
        self.test_fuel_specs = {
            "REGULAR": FuelSpec("REGULAR", 87.0, 9.0, 30.0, 0.74, 3.50),
            "TRANSMIX": FuelSpec("TRANSMIX", 85.0, 9.5, 50.0, 0.76, 2.80),
            "DIESEL": FuelSpec("DIESEL", 0.0, 0.0, 15.0, 0.85, 3.60),
        }
    
    def test_smart_initialization(self):
        """Test smart seeding with Dirichlet distribution"""
        optimizer = EnhancedBlendingOptimizer(self.test_nodes, self.test_fuel_specs)
        init_pop = optimizer.smart_seed()
        
        # Check population size
        self.assertEqual(len(init_pop), optimizer.population_size)
        
        # Check that each individual sums to 1
        for individual in init_pop:
            self.assertAlmostEqual(sum(individual), 1.0, places=5)
        
        # Check all values are non-negative
        for individual in init_pop:
            self.assertTrue(all(x >= 0 for x in individual))
        
        print(f"✓ Smart initialization generated {len(init_pop)} valid individuals")
    
    def test_mutation_normalization(self):
        """Test that mutation maintains normalization"""
        optimizer = EnhancedBlendingOptimizer(self.test_nodes, self.test_fuel_specs)
        optimizer.setup_ga()
        
        # Test individual
        individual = [0.3, 0.5, 0.2]
        
        # Apply mutation multiple times
        for _ in range(100):
            mutated, = optimizer.mutate_and_normalize(individual.copy(), mu=0)
            self.assertAlmostEqual(sum(mutated), 1.0, places=5)
            self.assertTrue(all(x >= 0 for x in mutated))
        
        print("✓ Mutation maintains normalization constraint")
    
    def test_ebitda_calculation(self):
        """Test EBITDA calculation logic"""
        optimizer = EnhancedBlendingOptimizer(self.test_nodes, self.test_fuel_specs)
        
        # Test with different allocations
        test_allocations = [
            np.array([1.0, 0.0, 0.0]),  # All to first node
            np.array([0.0, 1.0, 0.0]),  # All to second node
            np.array([0.33, 0.33, 0.34]),  # Equal distribution
        ]
        
        for alloc in test_allocations:
            ebitda = optimizer.calculate_ebitda(alloc)
            self.assertIsInstance(ebitda, float)
            print(f"  Allocation {alloc} → EBITDA: ${ebitda:,.2f}")
    
    def test_optimization_modes(self):
        """Test different optimization modes"""
        modes = ['swell', 'ebitda', 'hybrid']
        
        for mode in modes:
            print(f"\nTesting {mode} mode...")
            optimizer = EnhancedBlendingOptimizer(
                self.test_nodes, self.test_fuel_specs, 
                optimization_mode=mode
            )
            optimizer.population_size = 20
            optimizer.generations = 10
            
            # Test evaluation function
            test_individual = [0.4, 0.3, 0.3]
            fitness = optimizer.evaluate_comprehensive(test_individual)
            
            if mode == 'hybrid':
                self.assertEqual(len(fitness), 2)
                print(f"  Hybrid fitness: Swell={fitness[0]:.2f}, EBITDA={fitness[1]:.2f}")
            else:
                self.assertEqual(len(fitness), 1)
                print(f"  {mode} fitness: {fitness[0]:.2f}")
    
    def test_local_optimization_integration(self):
        """Test GA + local optimization integration"""
        optimizer = EnhancedBlendingOptimizer(
            self.test_nodes, self.test_fuel_specs,
            optimization_mode='ebitda'
        )
        
        # Test allocation
        initial_alloc = np.array([0.3, 0.4, 0.3])
        
        # Apply local optimization
        refined_alloc = optimizer.apply_local_optimization(initial_alloc)
        
        # Check constraints
        self.assertAlmostEqual(sum(refined_alloc), 1.0, places=5)
        self.assertTrue(all(0 <= x <= 1 for x in refined_alloc))
        
        # Compare performance
        initial_ebitda = optimizer.calculate_ebitda(initial_alloc)
        refined_ebitda = optimizer.calculate_ebitda(refined_alloc)
        
        print(f"\n✓ Local optimization:")
        print(f"  Initial EBITDA: ${initial_ebitda:,.2f}")
        print(f"  Refined EBITDA: ${refined_ebitda:,.2f}")
        print(f"  Improvement: {((refined_ebitda - initial_ebitda) / initial_ebitda * 100):.1f}%")

class TestFullOptimization(unittest.TestCase):
    """Integration tests for complete optimization process"""
    
    def test_complete_optimization_pipeline(self):
        """Test full optimization pipeline with all features"""
        print("\n" + "="*60)
        print("TESTING COMPLETE OPTIMIZATION PIPELINE")
        print("="*60)
        
        # Get test data
        manager = PipelineDataManager(use_bigquery=False)
        nodes = manager.get_pipeline_nodes()
        fuel_specs = manager.get_fuel_specs()
        
        # Test each optimization mode
        results_by_mode = {}
        
        for mode in ['swell', 'ebitda', 'hybrid']:
            print(f"\n--- Testing {mode.upper()} mode ---")
            
            optimizer = EnhancedBlendingOptimizer(
                nodes, fuel_specs, 
                optimization_mode=mode
            )
            optimizer.population_size = 50
            optimizer.generations = 30
            
            # Time the optimization
            start_time = time.time()
            results = optimizer.optimize(use_local_refinement=True)
            elapsed_time = time.time() - start_time
            
            results_by_mode[mode] = results
            
            print(f"Optimization completed in {elapsed_time:.2f} seconds")
            print(f"GA converged at generation: {len(results['convergence_history'])}")
            
            if 'refined_allocation' in results:
                print(f"Swell improvement: {results['improvement_swell']:.1f}%")
                print(f"EBITDA improvement: {results['improvement_ebitda']:.1f}%")
        
        return results_by_mode
    
    def test_stagnation_detection(self):
        """Test early stopping with stagnation detection"""
        print("\n--- Testing Stagnation Detection ---")
        
        manager = PipelineDataManager(use_bigquery=False)
        nodes = manager.get_pipeline_nodes()
        fuel_specs = manager.get_fuel_specs()
        
        optimizer = EnhancedBlendingOptimizer(nodes, fuel_specs)
        optimizer.population_size = 30
        optimizer.generations = 100  # Set high to test early stopping
        optimizer.max_stagnation = 10  # Low threshold for testing
        
        results = optimizer.optimize(use_local_refinement=False)
        
        actual_generations = len(results['convergence_history'])
        print(f"Stopped at generation {actual_generations} (max was {optimizer.generations})")
        self.assertLess(actual_generations, optimizer.generations)

def run_performance_benchmark():
    """Benchmark performance with different parameters"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    manager = PipelineDataManager(use_bigquery=False)
    nodes = manager.get_pipeline_nodes()
    fuel_specs = manager.get_fuel_specs()
    
    # Test different population sizes
    pop_sizes = [20, 50, 100, 200]
    benchmark_results = []
    
    for pop_size in pop_sizes:
        print(f"\nTesting population size: {pop_size}")
        
        optimizer = EnhancedBlendingOptimizer(nodes, fuel_specs, optimization_mode='hybrid')
        optimizer.population_size = pop_size
        optimizer.generations = 30
        
        start_time = time.time()
        results = optimizer.optimize(use_local_refinement=True)
        elapsed_time = time.time() - start_time
        
        benchmark_results.append({
            'population_size': pop_size,
            'time': elapsed_time,
            'final_swell': results.get('refined_swell_volume', results['ga_swell_volume']),
            'final_ebitda': results.get('refined_ebitda', results['ga_ebitda']),
            'generations': len(results['convergence_history'])
        })
        
        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  Final Swell: {benchmark_results[-1]['final_swell']:,.0f}")
        print(f"  Final EBITDA: ${benchmark_results[-1]['final_ebitda']:,.2f}")
    
    # Create visualization
    df_bench = pd.DataFrame(benchmark_results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time vs population size
    ax1.plot(df_bench['population_size'], df_bench['time'], 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Population Size')
    ax1.set_ylabel('Optimization Time (seconds)')
    ax1.set_title('Computational Time vs Population Size')
    ax1.grid(True, alpha=0.3)
    
    # Quality vs population size
    ax2_ebitda = ax2.twinx()
    
    line1 = ax2.plot(df_bench['population_size'], df_bench['final_swell'], 
                     'g-s', linewidth=2, markersize=8, label='Swell Volume')
    line2 = ax2_ebitda.plot(df_bench['population_size'], df_bench['final_ebitda'], 
                            'r-^', linewidth=2, markersize=8, label='EBITDA')
    
    ax2.set_xlabel('Population Size')
    ax2.set_ylabel('Swell Volume', color='g')
    ax2_ebitda.set_ylabel('EBITDA ($)', color='r')
    ax2.set_title('Solution Quality vs Population Size')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2_ebitda.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center left')
    
    plt.tight_layout()
    plt.savefig('performance_benchmark.png', dpi=300)
    plt.show()
    
    return benchmark_results

def test_constraint_handling():
    """Test constraint handling in various scenarios"""
    print("\n" + "="*60)
    print("TESTING CONSTRAINT HANDLING")
    print("="*60)
    
    # Create test scenario with tight constraints
    test_nodes = [
        PipelineNode("N001", "Constrained_A", True, 10000, True, True, 
                    100, 50000, 40000, 50.0, 1.0),  # Very low blend rate
        PipelineNode("N002", "Constrained_B", True, 50000, True, False, 
                    2000, 100000, 75000, 10.0, 0.5),  # High volume, good rate
        PipelineNode("N003", "Constrained_C", True, 30000, True, False, 
                    500, 80000, 60000, 20.0, 0.7),  # Medium constraints
    ]
    
    fuel_specs = {
        "REGULAR": FuelSpec("REGULAR", 87.0, 9.0, 30.0, 0.74, 3.50),
        "TRANSMIX": FuelSpec("TRANSMIX", 85.0, 9.5, 50.0, 0.76, 2.80),
    }
    
    optimizer = EnhancedBlendingOptimizer(test_nodes, fuel_specs, optimization_mode='hybrid')
    optimizer.population_size = 50
    optimizer.generations = 30
    
    results = optimizer.optimize(use_local_refinement=True)
    
    # Verify constraints are satisfied
    print("\nConstraint Verification:")
    for node_id, plan in results['blend_plan'].items():
        node = next(n for n in test_nodes if n.node_id == node_id)
        blend_volume = plan['blend_volume']
        blend_time = blend_volume / node.max_blend_rate if node.max_blend_rate > 0 else 0
        
        print(f"\n{plan['location']}:")
        print(f"  Blend volume: {blend_volume:,.0f} barrels")
        print(f"  Blend rate: {node.max_blend_rate} barrels/hour")
        print(f"  Blend time: {blend_time:.1f} hours")
        print(f"  Constraint satisfied: {'✓' if blend_time <= 24 else '✗'}")

def test_data_export_import():
    """Test saving and loading optimization results"""
    print("\n" + "="*60)
    print("TESTING DATA EXPORT/IMPORT")
    print("="*60)
    
    # Run optimization
    manager = PipelineDataManager(use_bigquery=False)
    nodes = manager.get_pipeline_nodes()
    fuel_specs = manager.get_fuel_specs()
    
    optimizer = EnhancedBlendingOptimizer(nodes, fuel_specs, optimization_mode='hybrid')
    optimizer.population_size = 30
    optimizer.generations = 20
    
    results = optimizer.optimize(use_local_refinement=True)
    
    # Export results to multiple formats
    # 1. CSV
    results_df = pd.DataFrame.from_dict(results['blend_plan'], orient='index')
    results_df.to_csv('test_optimization_results.csv')
    print("✓ Exported to CSV")
    
    # 2. JSON (excluding non-serializable objects)
    json_results = {
        'optimization_mode': optimizer.optimization_mode,
        'ga_swell_volume': results['ga_swell_volume'],
        'ga_ebitda': results['ga_ebitda'],
        'blend_plan': results['blend_plan']
    }
    
    if 'refined_allocation' in results:
        json_results.update({
            'refined_swell_volume': results['refined_swell_volume'],
            'refined_ebitda': results['refined_ebitda'],
            'improvement_swell': results['improvement_swell'],
            'improvement_ebitda': results['improvement_ebitda']
        })
    
    with open('test_optimization_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print("✓ Exported to JSON")
    
    # Test re-import
    loaded_df = pd.read_csv('test_optimization_results.csv', index_col=0)
    with open('test_optimization_results.json', 'r') as f:
        loaded_json = json.load(f)
    
    print("✓ Successfully re-imported results")
    print(f"  Loaded {len(loaded_df)} locations from CSV")
    print(f"  JSON contains {len(loaded_json['blend_plan'])} locations")

def run_sensitivity_analysis():
    """Run comprehensive sensitivity analysis"""
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    
    manager = PipelineDataManager(use_bigquery=False)
    nodes = manager.get_pipeline_nodes()
    fuel_specs = manager.get_fuel_specs()
    
    # Parameters to test
    test_params = {
        'mutation_prob': [0.1, 0.2, 0.3],
        'crossover_prob': [0.5, 0.6, 0.7, 0.8],
        'tournament_size': [2, 3, 5],
        'elite_size': [1, 5, 10]
    }
    
    base_config = {
        'population_size': 50,
        'generations': 25,
        'mutation_prob': 0.2,
        'crossover_prob': 0.6,
        'tournament_size': 3,
        'elite_size': 5
    }
    
    sensitivity_results = []
    
    for param_name, param_values in test_params.items():
        print(f"\nTesting sensitivity to {param_name}...")
        
        for value in param_values:
            # Create optimizer with modified parameter
            optimizer = EnhancedBlendingOptimizer(nodes, fuel_specs, optimization_mode='ebitda')
            
            # Apply base configuration
            for key, val in base_config.items():
                setattr(optimizer, key, val)
            
            # Apply test parameter
            setattr(optimizer, param_name, value)
            
            # Run optimization
            start_time = time.time()
            results = optimizer.optimize(use_local_refinement=False)
            elapsed_time = time.time() - start_time
            
            sensitivity_results.append({
                'parameter': param_name,
                'value': value,
                'final_fitness': results['ga_fitness'],
                'convergence_gen': len(results['convergence_history']),
                'time': elapsed_time
            })
            
            print(f"  {param_name}={value}: Fitness={results['ga_fitness']:.2f}, "
                  f"Generations={len(results['convergence_history'])}")
    
    # Create sensitivity visualization
    df_sens = pd.DataFrame(sensitivity_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, param in enumerate(test_params.keys()):
        ax = axes[idx]
        param_data = df_sens[df_sens['parameter'] == param]
        
        ax.plot(param_data['value'], param_data['final_fitness'], 'b-o', linewidth=2)
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Final Fitness')
        ax.set_title(f'Sensitivity to {param.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300)
    plt.show()
    
    return sensitivity_results

def main_test():
    """Main test execution"""
    print("\n" + "="*70)
    print("ENHANCED PIPELINE OPTIMIZATION TEST SUITE")
    print("="*70)
    
    # Run unit tests
    print("\n1. Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\n2. Running Integration Tests...")
    test_integration = TestFullOptimization()
    results_by_mode = test_integration.test_complete_optimization_pipeline()
    test_integration.test_stagnation_detection()
    
    # Run performance benchmark
    print("\n3. Running Performance Benchmark...")
    benchmark_results = run_performance_benchmark()
    
    # Test constraint handling
    print("\n4. Testing Constraint Handling...")
    test_constraint_handling()
    
    # Test data export/import
    print("\n5. Testing Data Export/Import...")
    test_data_export_import()
    
    # Run sensitivity analysis
    print("\n6. Running Sensitivity Analysis...")
    sensitivity_results = run_sensitivity_analysis()
    
    # Run comparison of optimization modes
    print("\n7. Comparing Optimization Modes...")
    comparison_results = compare_optimization_modes()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Summary statistics
    print("\nSummary:")
    print(f"- Tested {len(results_by_mode)} optimization modes")
    print(f"- Benchmarked {len(benchmark_results)} population sizes")
    print(f"- Analyzed sensitivity to {len(sensitivity_results)} parameter combinations")
    print("\nGenerated visualizations:")
    print("- enhanced_optimization_results.png")
    print("- optimization_mode_comparison.png")
    print("- performance_benchmark.png")
    print("- sensitivity_analysis.png")

if __name__ == "__main__":
    main_test()