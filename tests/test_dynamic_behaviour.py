import unittest
from dynamic_behaviour import *
from inputs.instance import DISTANCES_FILE, NODES_SHEET, VEHICLE_CAPACITY, read_instance_distance_matrix
from inputs.node import Node, InputNode, create_nodes_static
from simulated_annealing import objective, is_feasible

# The dynamic solution is stochastic, so the end-to-end tests are repeated. Each repetition draws a different sequence of moves out of the same problem instance
REPETITIONS = 25


def assert_valid_solution(test_case, tours: list[list[str]], nodes: list[Node]):
    """
    Assert that a finished solution serves every node that the revealed demand implies, exactly once, on tours that begin and end at the depot.

    Args:
        test_case: The test case making the assertion.
        tours (list[list[str]]): The tours of the finished solution.
        nodes (list[Node]): The nodes that exist once all demand has been revealed.

    """
    for tour in tours:
        test_case.assertEqual('0', tour[0], f"Tour {tour} does not start at the depot")
        test_case.assertEqual('0', tour[-1], f"Tour {tour} does not end at the depot")
        test_case.assertGreater(len(tour), 2, f"Tour {tour} serves no node")

    served = [node_id for tour in tours for node_id in tour if node_id != '0']
    test_case.assertEqual(len(served), len(set(served)), "A node is served more than once")
    test_case.assertEqual({node.id for node in nodes if node.id != '0'}, set(served), "The tours do not serve exactly the nodes of the revealed demand")


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
        node_families[1].child_nodes = [Node('1.1', 100), Node('1.2', 100), Node('1.3', 100)]

        reconcile_child_node_increase(ddm, dnl, nodes, unvisited_nodes, visited_node_family)

        # Only the child node the family has gained requires a visit. 1.1 and 1.2 were already part of the problem instance and are already accounted for
        self.assertEqual({'1.3'}, {node.id for node in unvisited_nodes})

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

        # The new child node inherits the distances of the node family it belongs to
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
        current_traversal_states = [['0'], ['0']]
        original_tour_positional_index = [0, 0]
        unvisited_nodes = set(nodes[1:])

        current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes, current_traversal_states, original_tour_positional_index = reconcile_child_node_decrease(
            current_tours, ddm, dnl, next_node_in_tour, nodes, original_tours, unvisited_nodes, visited_node_family, current_traversal_states, original_tour_positional_index)

        # The node that was just visited no longer exists, so the visit is attributed to the first child node of the family, which always does
        self.assertEqual('1.1', next_node_in_tour)
        self.assertEqual([['0', '1.1', '2.1', '0'], ['0', '2.2', '0']], current_tours)
        self.assertEqual([['0', '1.1', '2.1', '0'], ['0', '2.2', '0']], original_tours)
        self.assertEqual({'1.1', '2.1', '2.2'}, {node.id for node in unvisited_nodes})
        self.assertEqual(2, len(current_traversal_states))
        self.assertEqual(2, len(original_tour_positional_index))

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
        current_traversal_states = [['0'], ['0']]
        original_tour_positional_index = [0, 0]
        unvisited_nodes = set(nodes[1:])

        current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes, current_traversal_states, original_tour_positional_index = reconcile_child_node_decrease(
            current_tours, ddm, dnl, next_node_in_tour, nodes, original_tours, unvisited_nodes, visited_node_family, current_traversal_states, original_tour_positional_index)

        # Reattributing the visit empties the second tour, which must then be dropped along with its traversal state
        self.assertEqual('1.1', next_node_in_tour)
        self.assertEqual([['0', '1.1', '2.1', '0']], current_tours)
        self.assertEqual([['0', '1.1', '2.1', '0']], original_tours)
        self.assertEqual({'1.1', '2.1'}, {node.id for node in unvisited_nodes})
        self.assertEqual(1, len(current_traversal_states))
        self.assertEqual(1, len(original_tour_positional_index))

    def test_non_original_tours_may_empty_independently(self):
        # current_tours holds tours that are still waiting at the depot and have no counterpart in original_tours. One of those emptying is not a reconciliation failure
        np_distance_matrix = np.array([
            [0, 10, 15],
            [10, 0, 35],
            [15, 35, 0]
        ])
        input_nodes = [InputNode('0', 0, 0), InputNode('1', 150, 100), InputNode('2', 100, 100)]
        vehicle_capacity = 100

        ddm, dnl, node_families, nodes = initialise_dynamic_data_structures(np_distance_matrix, input_nodes, vehicle_capacity)
        visited_node_family = node_families[1]
        node_families[1].child_nodes = [node for node in node_families[1].child_nodes if node.id == '1.1']

        # The tour holding the doomed node 1.2 sits beyond the end of original_tours
        current_tours = [['0', '2.1', '0'], ['0', '1.1', '0'], ['0', '1.2', '0']]
        original_tours = [['0', '2.1', '0']]
        current_traversal_states = [['0'], ['0'], ['0']]
        original_tour_positional_index = [0]

        current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes, current_traversal_states, original_tour_positional_index = reconcile_child_node_decrease(
            current_tours, ddm, dnl, '2.1', nodes, original_tours, set(nodes[1:]), visited_node_family, current_traversal_states, original_tour_positional_index)

        self.assertEqual([['0', '2.1', '0'], ['0', '1.1', '0']], current_tours)
        self.assertEqual([['0', '2.1', '0']], original_tours)
        self.assertEqual(2, len(current_traversal_states))
        self.assertEqual(1, len(original_tour_positional_index))


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

        reordered_tours, reordered_traversal_states, original_tours = reconcile_new_and_current_sa_values(new_tours, new_traversal_states, original_tours)

        self.assertEqual(reordered_tours, [['0', '1.1', '0'], ['0', '3.2', '0'], ['0', '4.1', '0'], ['0', '2.1', '0'], ['0', '3.1', '0']])
        self.assertEqual(reordered_traversal_states, [['0', '1.1'], ['0', '3.2'], ['0', '4.1'], ['0', '2.1'], ['0', '3.1']])

        # The original tours must keep occupying the leading positions of the reordered list
        self.assertEqual(original_tours, reordered_tours[:len(original_tours)])

    def test_raises_when_an_original_tour_has_no_counterpart(self):
        original_tours = [['0', '9.9', '0']]
        new_tours = [['0', '1.1', '0']]
        new_traversal_states = [['0']]

        with self.assertRaises(IncorrectReconciliationError):
            reconcile_new_and_current_sa_values(new_tours, new_traversal_states, original_tours)


class TestUpdateOriginalTours(unittest.TestCase):
    def test_every_qualifying_tour_is_promoted(self):
        # The original tours occupy the leading positions of current_tours, which is the state reconcile_new_and_current_sa_values leaves behind. The tours waiting at the depot
        # follow, and those that have filled up must all be promoted, not merely the first of them
        nodes = [Node('0', 0)] + [Node(f'{i}.1', 900) for i in range(1, 8)] + [Node('9.1', 100)]
        original_tours = [['0', '1.1', '0'], ['0', '2.1', '0'], ['0', '3.1', '0']]
        original_tour_positional_index = [0, 0, 0]
        current_tours = copy.deepcopy(original_tours) + [['0', '9.1', '0'], ['0', '6.1', '0'], ['0', '7.1', '0']]
        current_traversal_states = [['0'] for _ in current_tours]

        update_original_tours(original_tours, original_tour_positional_index, current_tours, current_traversal_states, nodes, 1000, 0.5)

        self.assertEqual(5, len(original_tours))
        self.assertEqual([['0', '6.1', '0'], ['0', '7.1', '0']], original_tours[3:])
        self.assertEqual(5, len(original_tour_positional_index))

        # The tour that is only a tenth full stays at the depot
        self.assertNotIn(['0', '9.1', '0'], original_tours)

        # Promotion has to keep original_tours aligned with the leading positions of current_tours, because the traversal loop indexes both by the same counter
        self.assertEqual(original_tours, current_tours[:len(original_tours)])
        self.assertEqual(len(current_tours), len(current_traversal_states))

    def test_tours_below_the_utilisation_target_are_left_at_the_depot(self):
        nodes = [Node('0', 0), Node('1.1', 900), Node('2.1', 100)]
        original_tours = []
        original_tour_positional_index = []
        current_tours = [['0', '1.1', '0'], ['0', '2.1', '0']]
        current_traversal_states = [['0'], ['0']]

        update_original_tours(original_tours, original_tour_positional_index, current_tours, current_traversal_states, nodes, 1000, 0.5)

        self.assertEqual([['0', '1.1', '0']], original_tours)


class TestDynamicSA(unittest.TestCase):
    distance_matrix = np.array([
        [0, 10, 15, 20, 12],
        [10, 0, 35, 25, 44],
        [15, 35, 0, 30, 10],
        [20, 25, 30, 0, 4],
        [12, 44, 10, 4, 0]
    ])

    def run_repeatedly(self, nodes_factory, distance_matrix, initial_temp, iterations, vehicle_capacity, utilisation_target):
        """Run the dynamic solution repeatedly and assert that every run finishes on a valid solution."""
        for _ in range(REPETITIONS):
            current_tours_value, current_tours, execution_time, nodes = dynamic_sa(nodes_factory(), distance_matrix, objective, initial_temp, iterations, vehicle_capacity,
                                                                                   utilisation_target)

            self.assertFalse(np.isnan(current_tours_value), "The objective value is NaN")
            self.assertGreaterEqual(current_tours_value, 0)
            self.assertGreater(execution_time, 0)
            assert_valid_solution(self, current_tours, nodes)

    @staticmethod
    def simple_nodes():
        return [InputNode('0', 0, 0), InputNode('1', 1000, 10), InputNode('2', 400, 3300), InputNode('3', 700, 1000), InputNode('4', 200, 3000)]

    def test_simple(self):
        self.run_repeatedly(self.simple_nodes, self.distance_matrix, 10, 10, 600, 0.9)

    def test_simple_low_utilisation_target(self):
        self.run_repeatedly(self.simple_nodes, self.distance_matrix, 10, 5, 600, 0.5)

    def test_simple_no_utilisation_target(self):
        self.run_repeatedly(self.simple_nodes, self.distance_matrix, 10, 5, 600, 0)

    def test_single_node_low_expected_high_actual_demand(self):
        self.run_repeatedly(lambda: [InputNode('0', 0, 0), InputNode('1', 10, 6060)], np.array([[0, 10], [10, 0]]), 10, 5, 600, 0.9)

    def test_single_node_high_expected_low_actual_demand(self):
        self.run_repeatedly(lambda: [InputNode('0', 0, 0), InputNode('1', 10000, 1)], np.array([[0, 10], [10, 0]]), 10, 5, 600, 0.9)

    def test_simple_with_single_zero_actual(self):
        nodes_factory = lambda: [InputNode('0', 0, 0), InputNode('1', 1000, 10), InputNode('2', 400, 3300), InputNode('3', 700, 1000), InputNode('4', 200, 0)]
        self.run_repeatedly(nodes_factory, self.distance_matrix, 10, 5, 600, 0.9)

    def test_expected_demand_equals_actual_demand(self):
        # With no demand shocks the solution should never need to split a tour, and it must remain feasible against the nodes it started with
        nodes_factory = lambda: [InputNode('0', 0, 0), InputNode('1', 1000, 1000), InputNode('2', 400, 400), InputNode('3', 700, 700), InputNode('4', 200, 200)]
        self.run_repeatedly(nodes_factory, self.distance_matrix, 10, 10, 600, 0.9)

    def test_real(self):
        nodes = create_nodes_static(DISTANCES_FILE, NODES_SHEET)
        distance_matrix = read_instance_distance_matrix()

        current_tours_value, current_tours, execution_time, final_nodes = dynamic_sa(nodes, distance_matrix, objective, 100, 100, VEHICLE_CAPACITY, 0.9)

        assert_valid_solution(self, current_tours, final_nodes)
        self.assertTrue(is_feasible(current_tours, final_nodes, VEHICLE_CAPACITY))
        self.assertGreater(current_tours_value, 0)


if __name__ == '__main__':
    unittest.main()
