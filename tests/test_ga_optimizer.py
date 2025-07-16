import unittest
import numpy as np
from ga_optimizer.src.genetic_algorithm import smart_seed, evaluate_ebitda, mutate_and_normalize, run_genetic_algorithm
from ga_optimizer.src.utils import calculate_swell_volume, calculate_blending_cost
from ga_optimizer.src.main import apply_local_optimizer, Inputs

class TestGaOptimizer(unittest.TestCase):
    def setUp(self):
        self.inputs = Inputs()

    def test_smart_seed(self):
        pop_size = 50
        seeded_pop = smart_seed(pop_size, self.inputs.S_transmix, self.inputs.S_target, self.inputs.n_locations)
        self.assertEqual(len(seeded_pop), pop_size)
        for ind in seeded_pop:
            self.assertAlmostEqual(np.sum(ind), 1.0, places=5)

    def test_evaluate_ebitda(self):
        individual = [0.3, 0.4, 0.3]
        ebitda = evaluate_ebitda(individual, self.inputs.S_transmix, self.inputs.S_target, self.inputs.diesel_price, self.inputs.transport_cost_matrix, self.inputs.storage_cost_per_unit)
        self.assertIsInstance(ebitda, tuple)
        self.assertGreater(ebitda[0], 0)

    def test_mutate_and_normalize(self):
        individual = [0.3, 0.4, 0.3]
        mutated_individual, = mutate_and_normalize(individual)
        self.assertAlmostEqual(np.sum(mutated_individual), 1.0, places=5)

    def test_run_genetic_algorithm(self):
        best_qm, max_ebitda = run_genetic_algorithm(self.inputs, pop_size=50, ngen=10)
        self.assertAlmostEqual(np.sum(best_qm), 1.0, places=5)
        self.assertGreater(max_ebitda, 0)

    def test_apply_local_optimizer(self):
        initial_qm = [0.3, 0.4, 0.3]
        refined_qm = apply_local_optimizer('SLSQP', initial_qm, self.inputs)
        self.assertAlmostEqual(np.sum(refined_qm), 1.0, places=5)

    def test_calculate_swell_volume(self):
        swell_volume = calculate_swell_volume(100, self.inputs.S_transmix, self.inputs.S_target[0])
        self.assertEqual(swell_volume, 1.05)

    def test_calculate_blending_cost(self):
        blending_cost = calculate_blending_cost(100, self.inputs.S_target[0])
        self.assertEqual(blending_cost, 10)

if __name__ == '__main__':
    unittest.main()
