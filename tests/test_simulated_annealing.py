import unittest

import numpy as np

from errors.errors import InfeasibilityError, NodeNotFoundError
from inputs.dynamic_distance_matrix import DynamicDistanceMatrix
from inputs.node import InputNode, Node
from inputs.node_family import NodeFamily
from simulated_annealing import COOLING_SCHEDULES, create_boolean_matrix, get_cooling_schedule, get_metropolis_criterion, is_feasible, is_flow_conservation, \
    is_within_vehicle_capacity, objective, simulated_annealing, simulated_annealing_with_dynamic_constraints


def build_distance_matrix(input_nodes: list[InputNode], numpy_distance_matrix: np.array, vehicle_capacity: int) -> tuple:
    """Build an initialised DDM and its node list from a set of input nodes."""
    node_families = [NodeFamily(node, vehicle_capacity) for node in input_nodes]
    ddm = DynamicDistanceMatrix(numpy_distance_matrix, node_families)
    nodes = [node for node_family in node_families for node in node_family.child_nodes]
    ddm.initialise(nodes)
    return ddm, nodes


class TestCreateBooleanMatrix(unittest.TestCase):
    def test_single_tour(self):
        matrix = create_boolean_matrix([['0', '1.1', '0']])

        self.assertEqual(['0', '1.1'], list(matrix.index))
        self.assertEqual(1, matrix.at['0', '1.1'])
        self.assertEqual(1, matrix.at['1.1', '0'])
        self.assertEqual(0, matrix.at['0', '0'])

    def test_long_tours(self):
        tours = [['0', '1.1', '0'], ['0', '1.2', '2.1', '0'], ['0', '2.2', '2.3', '2.4', '0'], ['0', '3.1', '4.4', '0'], ['0', '1.5', '2.5', '0'],
                 ['0', '1.6', '4.4', '4.5', '5.1', '0'],
                 ['0', '5.2', '1.6', '2.6', '1.7', '1.8', '0'],
                 ['0', '5.3', '2.7', '2.8', '2.9', '5.4', '3.3', '0'],
                 ['0', '1.3', '3.2', '0'], ['0', '4.2', '4.3', '1.4', '0']]

        matrix = create_boolean_matrix(tours)

        # Every arc of every tour must be marked, and nothing else
        expected_arcs = {(tour[i], tour[i + 1]) for tour in tours for i in range(len(tour) - 1)}
        self.assertEqual(len(expected_arcs), int(matrix.values.sum()))
        for node1, node2 in expected_arcs:
            self.assertEqual(1, matrix.at[node1, node2])


class TestObjective(unittest.TestCase):
    def setUp(self):
        self.ddm, self.nodes = build_distance_matrix([InputNode('0', 0, 0), InputNode('1', 100, 100), InputNode('2', 100, 100)],
                                                     np.array([[0, 10, 15], [10, 0, 35], [15, 35, 0]]), 600)

    def test_objective_sums_the_arcs_of_the_tours(self):
        # Two return trips out of the depot: 2 x 10 to node 1.1 and 2 x 15 to node 2.1
        self.assertEqual(50, objective([['0', '1.1', '0'], ['0', '2.1', '0']], self.ddm))

    def test_objective_is_not_nan_when_a_node_is_left_unvisited(self):
        # Aligning the Boolean matrix with the DDM is what stops the unvisited node from turning the whole objective value into NaN
        value = objective([['0', '1.1', '0']], self.ddm)

        self.assertFalse(np.isnan(value))
        self.assertEqual(20, value)

    def test_objective_rejects_nodes_the_distance_matrix_does_not_know(self):
        with self.assertRaises(NodeNotFoundError):
            objective([['0', '9.9', '0']], self.ddm)


class TestFeasibility(unittest.TestCase):
    def setUp(self):
        self.nodes = [Node('0', 0), Node('1.1', 400), Node('2.1', 400)]

    def test_capacity_constraint_is_enforced_across_several_nodes(self):
        self.assertFalse(is_within_vehicle_capacity([['0', '1.1', '2.1', '0']], self.nodes, 600))
        self.assertTrue(is_within_vehicle_capacity([['0', '1.1', '0'], ['0', '2.1', '0']], self.nodes, 600))

    def test_a_single_node_tour_is_exempt_from_the_capacity_constraint(self):
        # A child node is one vehicle load and cannot be divided further, so its tour is feasible whatever its demand
        self.assertTrue(is_within_vehicle_capacity([['0', '1.1', '0']], [Node('0', 0), Node('1.1', 5000)], 600))

    def test_flow_conservation_rejects_a_node_visited_twice(self):
        self.assertFalse(is_flow_conservation([['0', '1.1', '0'], ['0', '1.1', '0']]))
        self.assertTrue(is_flow_conservation([['0', '1.1', '0'], ['0', '2.1', '0']]))

    def test_a_tour_that_does_not_return_to_the_depot_is_infeasible(self):
        self.assertFalse(is_feasible([['0', '1.1', '2.1']], self.nodes, 6000))


class TestCoolingSchedules(unittest.TestCase):
    def test_every_schedule_decreases_and_stays_positive(self):
        iterations = 1000

        for name in COOLING_SCHEDULES:
            cool = get_cooling_schedule(name)
            temperatures = [cool(100, i, iterations) for i in range(iterations)]

            self.assertAlmostEqual(100, temperatures[0], delta=1, msg=f"{name} does not start at the initial temperature")
            self.assertLess(temperatures[-1], temperatures[0], f"{name} does not cool")
            self.assertTrue(all(t > 0 for t in temperatures), f"{name} reaches a non-positive temperature")

    def test_unknown_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            get_cooling_schedule('geometric')

    def test_metropolis_criterion_is_bounded(self):
        # A worse candidate is accepted with a probability between 0 and 1 however extreme the difference or the temperature
        self.assertEqual(1, get_metropolis_criterion(0, 100))
        self.assertLess(get_metropolis_criterion(1000, 1), 1e-6)
        self.assertLessEqual(get_metropolis_criterion(-1000, 0), np.inf)
        self.assertGreaterEqual(get_metropolis_criterion(1000, 0), 0)


class TestSimulatedAnnealing(unittest.TestCase):
    def setUp(self):
        self.numpy_distance_matrix = np.array([
            [0, 10, 15, 20, 12],
            [10, 0, 35, 25, 44],
            [15, 35, 0, 30, 10],
            [20, 25, 30, 0, 4],
            [12, 44, 10, 4, 0]
        ])
        self.input_nodes = [InputNode('0', 0, 0), InputNode('1', 200, 200), InputNode('2', 200, 200), InputNode('3', 200, 200), InputNode('4', 200, 200)]
        self.ddm, self.nodes = build_distance_matrix(self.input_nodes, self.numpy_distance_matrix, 600)
        self.initial_tours = [['0', '1.1', '0'], ['0', '2.1', '0'], ['0', '3.1', '0'], ['0', '4.1', '0']]

    def test_infeasible_input_is_rejected(self):
        with self.assertRaises(InfeasibilityError):
            simulated_annealing([['0', '1.1', '2.1', '3.1', '4.1', '0']], self.nodes, self.ddm, objective, 10, 10, 600)

    def test_returned_value_matches_the_returned_tours(self):
        # The best value and the best tours are updated together, so recomputing the objective of the returned tours must reproduce the returned value
        for _ in range(25):
            value, tours = simulated_annealing(self.initial_tours, self.nodes, self.ddm, objective, 10, 50, 600)

            self.assertEqual(value, objective(tours, self.ddm))
            self.assertTrue(is_feasible(tours, self.nodes, 600))

    def test_the_result_never_worsens_the_initial_solution(self):
        initial_value = objective(self.initial_tours, self.ddm)

        for _ in range(25):
            value, tours = simulated_annealing(self.initial_tours, self.nodes, self.ddm, objective, 10, 50, 600)

            self.assertLessEqual(value, initial_value)

    def test_a_new_tour_can_be_created(self):
        # Extracting a node into the empty tour appended to the candidate list is what lets the search increase the number of tours
        merged_tours = [['0', '1.1', '2.1', '3.1', '0'], ['0', '4.1', '0']]
        created_more_tours = False

        for _ in range(50):
            _, tours = simulated_annealing(merged_tours, self.nodes, self.ddm, objective, 100, 50, 600)
            if len(tours) > len(merged_tours):
                created_more_tours = True
                break

        self.assertTrue(created_more_tours, "Simulated annealing never split a tour")


class TestSimulatedAnnealingWithDynamicConstraints(unittest.TestCase):
    def setUp(self):
        self.ddm, self.nodes = build_distance_matrix([InputNode('0', 0, 0), InputNode('1', 200, 200), InputNode('2', 200, 200), InputNode('3', 200, 200)],
                                                     np.array([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]), 600)
        self.tours = [['0', '1.1', '0'], ['0', '2.1', '0'], ['0', '3.1', '0']]

    def test_returns_tours_and_traversal_states(self):
        traversal_states = [['0'], ['0'], ['0']]

        value, tours, returned_traversal_states = simulated_annealing_with_dynamic_constraints(self.tours, self.nodes, self.ddm, objective, 10, 25, 600, traversal_states)

        self.assertEqual(value, objective(tours, self.ddm))
        self.assertEqual(len(tours), len(returned_traversal_states))

    def test_a_fully_traversed_solution_is_returned_unchanged(self):
        # Every tour has been driven to its end, so there is nothing left that may be moved
        traversal_states = [list(tour) for tour in self.tours]

        value, tours, returned_traversal_states = simulated_annealing_with_dynamic_constraints(self.tours, self.nodes, self.ddm, objective, 10, 25, 600, traversal_states)

        self.assertEqual(self.tours, tours)
        self.assertEqual(traversal_states, returned_traversal_states)

    def test_traversed_nodes_are_never_moved(self):
        traversal_states = [['0', '1.1'], ['0'], ['0']]

        for _ in range(25):
            _, tours, returned_traversal_states = simulated_annealing_with_dynamic_constraints(self.tours, self.nodes, self.ddm, objective, 10, 25, 600, traversal_states)

            for tour, traversal in zip(tours, returned_traversal_states):
                self.assertEqual(traversal, tour[:len(traversal)], "A node was inserted in front of an already traversed node")


if __name__ == '__main__':
    unittest.main()
