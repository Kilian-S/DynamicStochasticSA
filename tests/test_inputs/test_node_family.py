import unittest
from inputs.node import InputNode
from inputs.node_family import NodeFamily


class TestNodeFamily(unittest.TestCase):
    def test_update_expected_equals_actual_one_tour(self):
        input_node = InputNode('1', 600, 600)
        vehicle_capacity = 600
        node_family = NodeFamily(input_node, vehicle_capacity)

        self.assertEqual(1, len(node_family.child_nodes))

        node_family.update()

        self.assertEqual(1, len(node_family.child_nodes))
        self.assertEqual(0, node_family.child_nodes[0].expected_demand)

    def test_update_with_zero_actual(self):
        input_node = InputNode('1', 600, 0)
        vehicle_capacity = 600
        node_family = NodeFamily(input_node, vehicle_capacity)

        self.assertEqual(1, len(node_family.child_nodes))

        node_family.update()

        self.assertEqual(1, len(node_family.child_nodes))
        self.assertEqual(0, node_family.child_nodes[0].expected_demand)

    def test_update_expected_equals_actual_two_tour(self):
        input_node = InputNode('1', 1200, 1200)
        vehicle_capacity = 600
        node_family = NodeFamily(input_node, vehicle_capacity)

        self.assertEqual(2, len(node_family.child_nodes))

        node_family.update()

        self.assertEqual(2, len(node_family.child_nodes))
        self.assertEqual(600, node_family.child_nodes[0].expected_demand)
        self.assertEqual(600, node_family.child_nodes[1].expected_demand)

    def test_simple_expansion(self):
        input_node = InputNode('1', 1, 1500)
        vehicle_capacity = 600
        node_family = NodeFamily(input_node, vehicle_capacity)

        self.assertEqual(1, len(node_family.child_nodes))

        node_family.update()

        self.assertEqual(3, len(node_family.child_nodes))
        self.assertEqual(600, node_family.child_nodes[0].expected_demand)
        self.assertEqual(600, node_family.child_nodes[1].expected_demand)
        self.assertEqual(300, node_family.child_nodes[2].expected_demand)
        self.assertEqual('1.1', node_family.child_nodes[0].id)
        self.assertEqual('1.2', node_family.child_nodes[1].id)
        self.assertEqual('1.3', node_family.child_nodes[2].id)

    def test_simple_contraction(self):
        input_node = InputNode('1', 1500, 1)
        vehicle_capacity = 600
        node_family = NodeFamily(input_node, vehicle_capacity)

        self.assertEqual(3, len(node_family.child_nodes))
        self.assertEqual(600, node_family.child_nodes[0].expected_demand)
        self.assertEqual(600, node_family.child_nodes[1].expected_demand)
        self.assertEqual(300, node_family.child_nodes[2].expected_demand)
        self.assertEqual('1.1', node_family.child_nodes[0].id)
        self.assertEqual('1.2', node_family.child_nodes[1].id)
        self.assertEqual('1.3', node_family.child_nodes[2].id)

        node_family.update()

        self.assertEqual(1, len(node_family.child_nodes))
        self.assertEqual('1.1', node_family.child_nodes[0].id)


























