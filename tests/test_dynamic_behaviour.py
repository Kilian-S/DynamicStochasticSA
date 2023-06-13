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


class TestReconcileChildNodeIncrease(unittest.TestCase):
    def test_update_unvisited_nodes(self):
        np_distance_matrix = np.array([
            [0, 10, 15],
            [10, 0, 35],
            [15, 35, 0]
        ])

        input_nodes = [InputNode('0', 0, 0), InputNode('1', 150, 100), InputNode('2', 100, 100)]
        vehicle_capacity = 100

        ddm, dnl, node_families, nodes = initialise_dynamic_data_structures(np_distance_matrix, input_nodes, vehicle_capacity)
        visited_node_family = node_families[1]
        unvisited_nodes = set()
        node_families[1].child_nodes = [Node('1.1', 400), Node('1.2', 200)]

        updated_child_nodes = visited_node_family.child_nodes
        expected_unvisited_nodes = [[node for node in updated_child_nodes if node not in nodes]]

        reconcile_child_node_increase(ddm, dnl, nodes, unvisited_nodes, visited_node_family)

        self.assertEqual(unvisited_nodes, expected_unvisited_nodes)

    def test_update_ddm(self):
        np_distance_matrix = np.array([
            [0, 10, 15],
            [10, 0, 35],
            [15, 35, 0]
        ])
        input_nodes = [InputNode('0', 0, 0), InputNode('1', 100, 100), InputNode('2', 400, 400)]
        vehicle_capacity = 400

        ddm, dnl, node_families, nodes = initialise_dynamic_data_structures(np_distance_matrix, input_nodes, vehicle_capacity)
        visited_node_family = node_families[1]
        unvisited_nodes = set()
        node_families[1].child_nodes = [Node('1.1', 400), Node('1.2', 200)]
        reconcile_child_node_increase(ddm, dnl, nodes, unvisited_nodes, visited_node_family)

        self.assertEqual([10, 0, 0, 35], ddm.matrix.loc['1.2'].tolist())
        self.assertEqual([10, 0, 0, 35], ddm.matrix['1.2'].tolist())


class TestReconcileChildNodeDecrease(unittest.TestCase):
    def test_update_unvisited_nodes_without_empty_list_creation(self):
        np_distance_matrix = np.array([
            [0, 10, 15],
            [10, 0, 35],
            [15, 35, 0]
        ])
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
        np_distance_matrix = np.array([
            [0, 10, 15],
            [10, 0, 35],
            [15, 35, 0]
        ])
        input_nodes = [InputNode('0', 0, 0), InputNode('1', 150, 100), InputNode('2', 100, 100)]
        vehicle_capacity = 100

        ddm, dnl, node_families, nodes = initialise_dynamic_data_structures(np_distance_matrix, input_nodes, vehicle_capacity)
        next_node_in_tour = '1.2'
        visited_node_family = node_families[1]
        node_families[1].child_nodes = [node for node in node_families[1].child_nodes if node.id == '1.1']

        current_tours = [['0', '1.2', '2.1', '0'], ['0', '1.1', '0']]
        original_tours = copy.deepcopy(current_tours)
        unvisited_nodes = {'1.2', '2.1', '1.1'}

        current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes = reconcile_child_node_decrease(current_tours, ddm, dnl, next_node_in_tour, nodes, original_tours,
                                                                                                                 unvisited_nodes, visited_node_family)

        self.assertEqual('1.1', next_node_in_tour)
        self.assertEqual([['0', '1.1', '2.1', '0']], current_tours)
        self.assertEqual([['0', '1.1', '2.1', '0']], original_tours)
        self.assertEqual({'1.1', '2.1', }, set(unvisited_nodes))


class TestIntegrateNewlyAddedChildTours(unittest.TestCase):
    def test_integrate_newly_added_child_tours(self):
        np_distance_matrix = np.array([
            [0, 10, 15],
            [10, 0, 35],
            [15, 35, 0]
        ])
        input_nodes = [InputNode('0', 0, 0), InputNode('1', 100, 100), InputNode('2', 400, 400)]
        vehicle_capacity = 100

        ddm, dnl, node_families, nodes = initialise_dynamic_data_structures(np_distance_matrix, input_nodes, vehicle_capacity)
        tours = [['0', '1.1', '2.1', '0'], ['0', '2.2', '2.3', '2.4', '0']]
        traversal_states = [['0', '1.1'], ['0', '2.2', '2.3']]
        nodes.append(Node('2.5', 100))
        tours, traversal_states = integrate_newly_added_child_tours(tours, nodes, traversal_states)

        self.assertTrue(len(tours) == 3)
        self.assertTrue(tours[2] == ['0', '2.5', '0'])
        self.assertTrue(traversal_states[2] == ['0'])


class TestReconcileCurrentAndOriginalTours(unittest.TestCase):
    def test_reconcile_current_and_original_tours(self):
        original_tours = [['0', '1.1', '0'], ['0', '3.2', '0'], ['0', '4.1', '0']]
        new_tours = [['0', '2.1', '0'], ['0', '3.1', '0'], ['0', '3.2', '0'], ['0', '1.1', '0'], ['0', '4.1', '0']]
        new_traversal_states = [['0', '2.1'], ['0', '3.1'], ['0', '3.2'], ['0', '1.1'], ['0', '4.1']]

        reordered_tours, reordered_traversal_states = reconcile_new_and_current_sa_values(new_tours, new_traversal_states, original_tours)

        self.assertEqual(reordered_tours, [['0', '1.1', '0'], ['0', '3.2', '0'], ['0', '4.1', '0'], ['0', '2.1', '0'], ['0', '3.1', '0']])
        self.assertEqual(reordered_traversal_states, [['0', '1.1'], ['0', '3.2'], ['0', '4.1'], ['0', '2.1'], ['0', '3.1']])


class TestDynamicSA(unittest.TestCase):
    def test_simple(self):
        initial_temp = 10
        iterations = 5
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

        while True:
            dynamic_sa(nodes, distance_matrix, objective, initial_temp, iterations, vehicle_capacity, utilisation_target)

    def test_simple_low_utilisation_target(self):
        initial_temp = 10
        iterations = 5
        utilisation_target = 0.5
        vehicle_capacity = 600
        nodes = [InputNode('0', 0, 0), InputNode('1', 1000, 10), InputNode('2', 400, 3300), InputNode('3', 700, 1000), InputNode('4', 200, 3000)]

        distance_matrix = np.array([
            [0, 10, 15, 20, 12],
            [10, 0, 35, 25, 44],
            [15, 35, 0, 30, 10],
            [20, 25, 30, 0, 4],
            [12, 44, 10, 4, 0]
        ])

        while True:
            dynamic_sa(nodes, distance_matrix, objective, initial_temp, iterations, vehicle_capacity, utilisation_target)

    def test_simple_no_utilisation_target(self):
        initial_temp = 10
        iterations = 5
        utilisation_target = 0
        vehicle_capacity = 600
        nodes = [InputNode('0', 0, 0), InputNode('1', 1000, 10), InputNode('2', 400, 3300), InputNode('3', 700, 1000), InputNode('4', 200, 3000)]

        distance_matrix = np.array([
            [0, 10, 15, 20, 12],
            [10, 0, 35, 25, 44],
            [15, 35, 0, 30, 10],
            [20, 25, 30, 0, 4],
            [12, 44, 10, 4, 0]
        ])

        while True:
            dynamic_sa(nodes, distance_matrix, objective, initial_temp, iterations, vehicle_capacity, utilisation_target)

    def test_single_node_low_expected_high_actual_demand(self):
        initial_temp = 10
        iterations = 5
        utilisation_target = 0.9
        vehicle_capacity = 600
        nodes = [InputNode('0', 0, 0), InputNode('1', 10, 6060)]

        distance_matrix = np.array([
            [0, 10],
            [10, 0]
        ])

        while True:
            dynamic_sa(nodes, distance_matrix, objective, initial_temp, iterations, vehicle_capacity, utilisation_target)

    def test_single_node_high_expected_low_actual_demand(self):
        initial_temp = 10
        iterations = 5
        utilisation_target = 0.9
        vehicle_capacity = 600
        nodes = [InputNode('0', 0, 0), InputNode('1', 10000, 1)]

        distance_matrix = np.array([
            [0, 10],
            [10, 0]
        ])

        dynamic_sa(nodes, distance_matrix, objective, initial_temp, iterations, vehicle_capacity, utilisation_target)

    def test_simple_with_single_zero_actual(self):
        initial_temp = 10
        iterations = 5
        utilisation_target = 0.9
        vehicle_capacity = 600
        nodes = [InputNode('0', 0, 0), InputNode('1', 1000, 10), InputNode('2', 400, 3300), InputNode('3', 700, 1000), InputNode('4', 200, 0)]

        distance_matrix = np.array([
            [0, 10, 15, 20, 12],
            [10, 0, 35, 25, 44],
            [15, 35, 0, 30, 10],
            [20, 25, 30, 0, 4],
            [12, 44, 10, 4, 0]
        ])

        while True:
            dynamic_sa(nodes, distance_matrix, objective, initial_temp, iterations, vehicle_capacity, utilisation_target)

    def test_real(self):




























