import unittest
import numpy as np

from ga_optimizer.src.genetic_algorithm import smart_seed, evaluate_ebitda, mutate_and_normalize, run_genetic_algorithm
from ga_optimizer.src.main import Inputs
from deap import creator, base

class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        self.inputs = Inputs()
        # Create fitness and individual classes if they don't exist
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)


    def test_smart_seed(self):
        pop_size = 10
        init_qms = smart_seed(pop_size, self.inputs.S_transmix, self.inputs.S_target, self.inputs.n_locations)
        self.assertEqual(len(init_qms), pop_size)
        for qm in init_qms:
            self.assertAlmostEqual(np.sum(qm), 1.0)

    def test_evaluate_ebitda(self):
        individual = [0.3, 0.4, 0.3]
        ebitda = evaluate_ebitda(individual, self.inputs.S_transmix, self.inputs.S_target, self.inputs.diesel_price, self.inputs.transport_cost_matrix, self.inputs.storage_cost_per_unit)
        self.assertIsInstance(ebitda, tuple)
        self.assertEqual(len(ebitda), 1)

    def test_mutate_and_normalize(self):
        individual = creator.Individual([0.3, 0.4, 0.3])
        mutated_individual, = mutate_and_normalize(individual)
        self.assertAlmostEqual(np.sum(mutated_individual), 1.0)

    def test_run_genetic_algorithm(self):
        best_qm, max_ebitda = run_genetic_algorithm(self.inputs)
        self.assertAlmostEqual(np.sum(best_qm), 1.0)
        self.assertGreater(max_ebitda, 0)

if __name__ == '__main__':
    unittest.main()
