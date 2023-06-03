from typing import List

from inputs.node import Node, InputNode


class NodeFamily:
    def __init__(self, input_node: InputNode, vehicle_capacity: int, is_visited=False):
        self.node_family_id = input_node.id
        self.node_family_expected_demand = input_node.expected_demand
        self.node_family_actual_demand = input_node.actual_demand
        self.vehicle_capacity = vehicle_capacity
        self.child_nodes: List[Node] = []
        self.is_visited = is_visited

        self.initialise()

    def create_child_nodes(self, trips_required, remaining_capacity):
        for i in range(trips_required):
            new_node = Node(f'{self.node_family_id}.{i + 1}', self.vehicle_capacity)
            self.child_nodes.append(new_node)

        if remaining_capacity > 0:
            remaining_node = Node(f'{self.node_family_id}.{trips_required + 1}', remaining_capacity)
            self.child_nodes.append(remaining_node)

    def initialise(self):
        if self.node_family_expected_demand == 0:
            if self.node_family_id == '0':
                self.child_nodes = [Node('0', 0)]
                self.is_visited = True
            else:
                self.child_nodes = [Node(f'{self.node_family_id}.{1}', 0)]

        expected_trips_required = self.node_family_expected_demand // self.vehicle_capacity
        remaining_capacity = self.node_family_expected_demand % self.vehicle_capacity
        self.create_child_nodes(expected_trips_required, remaining_capacity)

    def update(self):
        self.is_visited = True
        expected_trips_required = self.node_family_expected_demand // self.vehicle_capacity
        actual_trips_required = self.node_family_actual_demand // self.vehicle_capacity
        remaining_capacity = self.node_family_actual_demand % self.vehicle_capacity

        if actual_trips_required == expected_trips_required:
            if remaining_capacity == 0:
                # Delete last node in child_nodes
                self.child_nodes.pop()
            else:
                # Update demand of last node in child_nodes
                self.child_nodes[-1].expected_demand = self.node_family_actual_demand % self.vehicle_capacity

        else:
            # Fewer / more trips are required than planned: self.child_nodes must shrink / grow
            # Clear existing child_nodes list
            self.child_nodes.clear()
            self.create_child_nodes(actual_trips_required, remaining_capacity)

    def __repr__(self):
        return f"Node family {self.node_family_id}, Child nodes: {self.child_nodes}"


