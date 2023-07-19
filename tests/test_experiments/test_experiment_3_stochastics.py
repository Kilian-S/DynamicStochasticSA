import unittest

import numpy as np

from dynamic_behaviour import dynamic_sa
from experiments.experiment_3_stochastics.stochastics import get_omniscient_nodes
from inputs.node import InputNode
from simulated_annealing import objective


class TestStochastics(unittest.TestCase):
    def test_omniscient_nodes_simple(self):
        initial_temp = 10
        iterations = 10
        utilisation_target = 0.9
        vehicle_capacity = 600
        nodes = [InputNode('0', 0, 0), InputNode('1', 1000, 10), InputNode('2', 400, 3300), InputNode('3', 700, 1000), InputNode('4', 200, 3000)]

        distance_matrix = np.array([
            [0, 10, 15, 20, 12],
            [10, 0, 35, 25, 44],
            [15, 35, 0, 30, 10],
            [20, 25, 30, 0, 4],
            [12, 44, 10, 4, 0]
        ])

        current_tours_value, current_tours, execution_time, *_ = dynamic_sa(nodes, distance_matrix, objective, initial_temp, iterations, vehicle_capacity, utilisation_target)
        current_tours_set = set(string for tour in current_tours for string in tour)

        omniscient_nodes = get_omniscient_nodes(nodes, vehicle_capacity)
        omniscient_nodes_set = set(node.id for node in omniscient_nodes)

        self.assertEqual(current_tours_set, omniscient_nodes_set)





