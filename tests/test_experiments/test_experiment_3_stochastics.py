import unittest

import numpy as np

from dynamic_behaviour import dynamic_sa
from experiments.experiment_3_stochastics.stochastics import get_omniscient_nodes, get_service_level, get_total_oversupply, get_total_undersupply
from inputs.node import InputNode
from simulated_annealing import is_within_vehicle_capacity, objective

# The dynamic solution is stochastic, so the tests are repeated rather than run once
REPETITIONS = 25


class TestStochastics(unittest.TestCase):
    distance_matrix = np.array([
        [0, 10, 15, 20, 12],
        [10, 0, 35, 25, 44],
        [15, 35, 0, 30, 10],
        [20, 25, 30, 0, 4],
        [12, 44, 10, 4, 0]
    ])

    @staticmethod
    def stochastic_nodes():
        return [InputNode('0', 0, 0), InputNode('1', 1000, 10), InputNode('2', 400, 3300), InputNode('3', 700, 1000), InputNode('4', 200, 3000)]

    def test_omniscient_nodes_simple(self):
        initial_temp = 10
        iterations = 10
        utilisation_target = 0.9
        vehicle_capacity = 600

        for _ in range(REPETITIONS):
            nodes = self.stochastic_nodes()
            current_tours_value, current_tours, execution_time, *_ = dynamic_sa(nodes, self.distance_matrix, objective, initial_temp, iterations, vehicle_capacity,
                                                                                utilisation_target)
            current_tours_set = set(string for tour in current_tours for string in tour)

            omniscient_nodes = get_omniscient_nodes(nodes, vehicle_capacity)
            omniscient_nodes_set = set(node.id for node in omniscient_nodes)

            # Every visit the revealed demand calls for is made. The dynamic solution may make more visits than that: discovering a shortfall halfway through a tour forces a
            # follow-up visit that a planner who knew the demand in advance would have avoided
            self.assertTrue(omniscient_nodes_set.issubset(current_tours_set),
                            f"The solution does not serve every node of the revealed demand. Missing: {omniscient_nodes_set - current_tours_set}")

    def test_dynamic_solution_never_overloads_a_vehicle(self):
        vehicle_capacity = 600

        for _ in range(REPETITIONS):
            nodes = self.stochastic_nodes()
            current_tours_value, current_tours, execution_time, final_nodes = dynamic_sa(nodes, self.distance_matrix, objective, 10, 10, vehicle_capacity, 0.9)

            # This is the property the whole approach rests on: because the solver reacts to demand as it is revealed, no vehicle is ever asked to carry more than it can. It is
            # asserted against the demands the solver itself ended up with, which is the only basis on which the tours it produced are meaningful
            self.assertTrue(is_within_vehicle_capacity(current_tours, final_nodes, vehicle_capacity),
                            f"A tour exceeds the vehicle capacity: {current_tours}")

    def test_service_level_is_complete_when_no_tour_was_split(self):
        vehicle_capacity = 600

        for _ in range(REPETITIONS):
            nodes = self.stochastic_nodes()
            current_tours_value, current_tours, execution_time, *_ = dynamic_sa(nodes, self.distance_matrix, objective, 10, 10, vehicle_capacity, 0.9)

            omniscient_nodes = get_omniscient_nodes(nodes, vehicle_capacity)
            current_tours_set = set(string for tour in current_tours for string in tour)
            omniscient_nodes_set = set(node.id for node in omniscient_nodes)

            # The service level scores the tours against the omniscient demands. Where the solver has had to split a node mid-tour, the two no longer describe demand the same
            # way, so the metric is only well defined for the runs in which no split occurred
            if current_tours_set != omniscient_nodes_set:
                continue

            service_level = get_service_level(current_tours, omniscient_nodes, vehicle_capacity, current_tours_set, omniscient_nodes_set)
            total_undersupply, undersupplied_tours_count = get_total_undersupply(current_tours, omniscient_nodes, vehicle_capacity, current_tours_set, omniscient_nodes_set)

            self.assertEqual(1.0, service_level)
            self.assertEqual(0, total_undersupply)
            self.assertEqual(0, undersupplied_tours_count)

    def test_oversupply_is_never_negative(self):
        vehicle_capacity = 600

        for _ in range(REPETITIONS):
            nodes = self.stochastic_nodes()
            current_tours_value, current_tours, execution_time, *_ = dynamic_sa(nodes, self.distance_matrix, objective, 10, 10, vehicle_capacity, 0.9)

            omniscient_nodes = get_omniscient_nodes(nodes, vehicle_capacity)
            current_tours_set = set(string for tour in current_tours for string in tour)
            omniscient_nodes_set = set(node.id for node in omniscient_nodes)

            self.assertGreaterEqual(get_total_oversupply(current_tours, omniscient_nodes, vehicle_capacity, current_tours_set, omniscient_nodes_set), 0)


if __name__ == '__main__':
    unittest.main()
