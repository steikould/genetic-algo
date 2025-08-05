# Contains utility functions for creating visualizations.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict

def visualize_optimization_results(results: Dict, optimizer):
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

def compare_optimization_modes(results_comparison, optimizer):
    """Compare different optimization modes"""
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
