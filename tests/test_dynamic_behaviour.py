import unittest
import numpy as np
import copy

from dynamic_behaviour import *
from global_parameters import *
from inputs.dynamic_distance_matrix import DynamicDistanceMatrix, get_node_family_from_child_node
from inputs.dynamic_nodes_list import DynamicNodeList
from inputs.node import Node
from inputs.node_family import NodeFamily
from simulated_annealing import objective, simulated_annealing, simulated_annealing_with_dynamic_constraints
from inputs.node import InputNode

np_distance_matrix = np.array([
    [0, 10, 15],
    [10, 0, 35],
    [15, 35, 0]
])

input_nodes = [InputNode('0', 0, 0), InputNode('1', 150, 100), InputNode('2', 100, 100)]
vehicle_capacity = 100


class TestReconcileChildNodeIncrease(unittest.TestCase):
    def test_update_unvisited_nodes(self):
        ddm, dnl, node_families, nodes = initialise_dynamic_data_structures(np_distance_matrix, input_nodes, vehicle_capacity)
        visited_node_family = node_families[1]
        unvisited_nodes = []
        node_families[1].child_nodes = [Node('1.1', 400), Node('1.2', 200)]

        updated_child_nodes = visited_node_family.child_nodes
        expected_unvisited_nodes = [[node for node in updated_child_nodes if node not in nodes]]

        reconcile_child_node_increase(ddm, dnl, nodes, unvisited_nodes, visited_node_family)

        self.assertEqual(unvisited_nodes, expected_unvisited_nodes)

    def test_update_ddm(self):
        input_nodes = [InputNode('0', 0, 0), InputNode('1', 100, 100), InputNode('2', 400, 400)]
        vehicle_capacity = 400

        ddm, dnl, node_families, nodes = initialise_dynamic_data_structures(np_distance_matrix, input_nodes, vehicle_capacity)
        visited_node_family = node_families[1]
        unvisited_nodes = []
        node_families[1].child_nodes = [Node('1.1', 400), Node('1.2', 200)]
        reconcile_child_node_increase(ddm, dnl, nodes, unvisited_nodes, visited_node_family)

        self.assertEqual([10, 0, 0, 35], ddm.matrix.loc['1.2'].tolist())
        self.assertEqual([10, 0, 0, 35], ddm.matrix['1.2'].tolist())


class TestReconcileChildNodeDecrease(unittest.TestCase):
    def test_update_unvisited_nodes_without_empty_list_creation(self):
        input_nodes = [InputNode('0', 0, 0), InputNode('1', 150, 100), InputNode('2', 200, 100)]
        vehicle_capacity = 100

        ddm, dnl, node_families, nodes = initialise_dynamic_data_structures(np_distance_matrix, input_nodes, vehicle_capacity)
        next_node_in_tour = '1.2'
        visited_node_family = node_families[1]
        node_families[1].child_nodes = [node for node in node_families[1].child_nodes if node.id == '1.1']

        current_tours = [['0', '1.2', '2.1', '0'], ['0', '1.1', '2.2', '0']]
        original_tours = copy.deepcopy(current_tours)
        unvisited_nodes = {'1.2', '2.1', '1.1', '2.2'}

        current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes = reconcile_child_node_decrease(current_tours, ddm, dnl, next_node_in_tour, nodes, original_tours,
                                                                                                               unvisited_nodes, visited_node_family)

        self.assertEqual('1.1', next_node_in_tour)
        self.assertEqual([['0', '1.1', '2.1', '0'], ['0', '2.2', '0']], current_tours)
        self.assertEqual([['0', '1.1', '2.1', '0'], ['0', '2.2', '0']], original_tours)
        self.assertEqual({'1.1', '2.1', '2.2'}, set(unvisited_nodes))

    def test_update_unvisited_nodes_with_empty_list_creation(self):
        input_nodes = [InputNode('0', 0, 0), InputNode('1', 150, 100), InputNode('2', 100, 100)]
        vehicle_capacity = 100

        ddm, dnl, node_families, nodes = initialise_dynamic_data_structures(np_distance_matrix, input_nodes, vehicle_capacity)
        next_node_in_tour = '1.2'
        visited_node_family = node_families[1]
        node_families[1].child_nodes = [node for node in node_families[1].child_nodes if node.id == '1.1']

        current_tours = [['0', '1.2', '2.1', '0'], ['0', '1.1', '0']]
        original_tours = copy.deepcopy(current_tours)
        unvisited_nodes = ['1.2', '2.1', '1.1']

        current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes = reconcile_child_node_decrease(current_tours, ddm, dnl, next_node_in_tour, nodes, original_tours,
                                                                                                               unvisited_nodes, visited_node_family)

        self.assertEqual('1.1', next_node_in_tour)
        self.assertEqual([['0', '1.1', '2.1', '0']], current_tours)
        self.assertEqual([['0', '1.1', '2.1', '0']], original_tours)
        self.assertEqual({'1.1', '2.1',}, set(unvisited_nodes))













































