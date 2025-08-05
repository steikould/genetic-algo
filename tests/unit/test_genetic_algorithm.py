import unittest
import numpy as np

from ga_optimizer.initialization.smart_seeding import smart_seed
from ga_optimizer.evaluation.fuel_evaluators import calculate_ebitda, evaluate_comprehensive
from ga_optimizer.algorithms.genetic.operators import mutate_and_normalize
from ga_optimizer.algorithms.genetic.deap_ga import run_genetic_algorithm
from ga_optimizer.data.providers.synthetic import Inputs, PipelineDataManager
from deap import creator, base

class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        self.inputs = Inputs()
        self.data_manager = PipelineDataManager()
        self.blending_nodes = self.data_manager.get_pipeline_nodes()
        self.fuel_specs = self.data_manager.get_fuel_specs()
        # Create fitness and individual classes if they don't exist
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)


    def test_smart_seed(self):
        pop_size = 10
        center = [1/3, 1/3, 1/3]
        init_qms = smart_seed(pop_size, center, self.inputs.S_transmix, self.inputs.S_target, self.inputs.n_locations)
        self.assertEqual(len(init_qms), pop_size)
        for qm in init_qms:
            self.assertAlmostEqual(np.sum(qm), 1.0, places=5)

    def test_calculate_ebitda(self):
        individual = [0.3, 0.4, 0.3]
        ebitda = calculate_ebitda(np.array(individual), self.blending_nodes, self.fuel_specs)
        self.assertIsInstance(ebitda, float)

    def test_evaluate_comprehensive(self):
        individual = [0.3, 0.4, 0.3]
        fitness = evaluate_comprehensive(individual, self.blending_nodes, self.fuel_specs, 'ebitda')
        self.assertIsInstance(fitness, tuple)
        self.assertEqual(len(fitness), 1)

    def test_mutate_and_normalize(self):
        individual = creator.Individual([0.3, 0.4, 0.3])
        mutated_individual, = mutate_and_normalize(individual)
        self.assertAlmostEqual(np.sum(mutated_individual), 1.0, places=5)

    def test_run_genetic_algorithm(self):
        results = run_genetic_algorithm(self.inputs)
        best_qm = results["best_qm"]
        max_ebitda = results["final_fitness"]
        self.assertAlmostEqual(np.sum(best_qm), 1.0, places=5)
        self.assertGreater(max_ebitda, 0)

if __name__ == '__main__':
    unittest.main()
