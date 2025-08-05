# Domain entities (moved from current models.py)
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

class ExecutionContext(BaseModel):
    """Captures the complete optimization environment"""

    # Core Configuration
    algorithm_config: Dict[str, Any] = Field(..., description="All algorithm parameters")
    problem_characteristics: Dict[str, float] = Field(..., description="Problem features")
    hyperparameters: Dict[str, Any] = Field(..., description="Tunable parameters")
    hyperparameter_source: str = Field(..., description="How hyperparams were selected")

    # Environmental Context
    execution_timestamp: datetime = Field(default_factory=datetime.now)
    computational_resources: Dict[str, Any] = Field(..., description="Hardware/time constraints")
    random_seed: Optional[int] = None
    external_conditions: Dict[str, Any] = Field(default_factory=dict)

    # Problem Context
    problem_domain: str = Field(..., description="Energy, logistics, scheduling, etc.")
    problem_size: Dict[str, int] = Field(..., description="Variables, constraints, etc.")
    problem_difficulty: Optional[float] = None
    similar_problems: List[str] = Field(default_factory=list)

    @property
    def context_fingerprint(self) -> str:
        """Unique identifier for this execution context"""
        # Convert dict to a frozenset of items for stable hashing
        algo_config_hash = hash(frozenset(self.algorithm_config.items()))
        problem_chars_hash = hash(frozenset(self.problem_characteristics.items()))
        return hash(f"{algo_config_hash}{problem_chars_hash}")

class PerformanceMetrics(BaseModel):
    """Detailed performance tracking throughout optimization"""

    # Convergence Data
    fitness_trajectory: List[float] = Field(..., description="Best fitness per generation")
    diversity_trajectory: List[float] = Field(..., description="Population diversity over time")
    convergence_rate: float = Field(..., description="Speed of convergence")
    convergence_stability: float = Field(..., description="Stability of convergence")

    # Resource Utilization
    execution_time: float = Field(..., description="Total runtime")
    evaluations_count: int = Field(..., description="Total fitness evaluations")
    memory_peak: float = Field(..., description="Peak memory usage")
    efficiency_score: float = Field(..., description="Quality per unit time")

    # Quality Metrics
    final_fitness: float = Field(..., description="Best achieved fitness")
    solution_quality: float = Field(..., description="Normalized quality score")
    robustness_score: float = Field(..., description="Solution stability")
    constraint_satisfaction: float = Field(..., description="Constraint adherence")

    # Comparative Metrics
    baseline_improvement: float = Field(..., description="Improvement over baseline")
    theoretical_optimum_gap: Optional[float] = None
    previous_best_improvement: Optional[float] = None

class PopulationDynamics(BaseModel):
    """Captures how the population evolved during optimization"""

    # Population Evolution
    population_snapshots: List[Dict[str, Any]] = Field(..., description="Key generation snapshots")
    genetic_diversity_evolution: List[float] = Field(..., description="Diversity over time")
    elite_preservation_rate: List[float] = Field(..., description="Elite retention per generation")

    # Operator Effectiveness
    crossover_success_rates: List[float] = Field(..., description="Crossover effectiveness per generation")
    mutation_impact_scores: List[float] = Field(..., description="Mutation effectiveness per generation")
    selection_pressure_evolution: List[float] = Field(..., description="Selection pressure over time")

    # Population Health
    population_stagnation_periods: List[tuple] = Field(..., description="Periods of no improvement")
    breakthrough_generations: List[int] = Field(..., description="Generations with significant improvement")
    population_clustering_analysis: Dict[str, Any] = Field(..., description="Population structure analysis")

class StrategicInsights(BaseModel):
    """High-level insights about what worked and why"""

    # Effectiveness Analysis
    most_effective_operators: Dict[str, float] = Field(..., description="Operator effectiveness ranking")
    optimal_parameter_ranges: Dict[str, tuple] = Field(..., description="Effective parameter ranges")
    critical_success_factors: List[str] = Field(..., description="Key factors for success")
    failure_modes: List[str] = Field(..., description="What caused failures")

    # Strategic Recommendations
    recommended_strategies: List[str] = Field(..., description="Strategies for similar problems")
    parameter_sensitivity_analysis: Dict[str, float] = Field(..., description="Parameter importance")
    transfer_learning_potential: float = Field(..., description="How transferable are insights")

    # Causal Analysis
    causal_relationships: Dict[str, List[str]] = Field(..., description="X causes Y relationships")
    intervention_effects: Dict[str, float] = Field(..., description="Effect of parameter changes")
    counterfactual_scenarios: List[Dict[str, Any]] = Field(..., description="What-if analyses")

class TransferabilityMetadata(BaseModel):
    """Information enabling knowledge transfer to new problems"""

    # Similarity Metrics
    problem_similarity_features: Dict[str, float] = Field(..., description="Features for similarity comparison")
    generalization_boundaries: Dict[str, tuple] = Field(..., description="Where insights apply")
    transfer_confidence: float = Field(..., description="Confidence in transferring insights")

    # Transfer Learning Weights
    feature_importance_weights: Dict[str, float] = Field(..., description="Feature importance for transfer")
    context_adaptation_rules: List[str] = Field(..., description="How to adapt to new contexts")
    transfer_learning_coefficients: Dict[str, float] = Field(..., description="Transfer learning parameters")

    # Validation Data
    cross_validation_scores: List[float] = Field(..., description="Validation of transferability")
    successful_transfers: List[str] = Field(..., description="Where this knowledge was successfully applied")
    failed_transfers: List[str] = Field(..., description="Where transfer failed and why")

class OptimizationResult(BaseModel):
    """The complete results of an optimization run"""

    execution_context: ExecutionContext
    performance_metrics: PerformanceMetrics
    population_dynamics: PopulationDynamics
    strategic_insights: StrategicInsights
    transferability_metadata: TransferabilityMetadata
