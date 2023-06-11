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

    def create_child_nodes(self, trips_required: int, remaining_capacity: int):
        for i in range(trips_required):
            new_node = Node(f'{self.node_family_id}.{i + 1}', self.vehicle_capacity)
            self.child_nodes.append(new_node)

        if remaining_capacity > 0:
            remaining_node = Node(f'{self.node_family_id}.{trips_required + 1}', remaining_capacity)
            self.child_nodes.append(remaining_node)

    def create_new_child_node(self, expected_demand: int):
        highest_id = max(int(node.id.split('.')[1]) for node in self.child_nodes)
        new_id = self.node_family_id + "." + str(highest_id + 1)

        new_node = Node(new_id, expected_demand)
        self.child_nodes.append(new_node)

        return new_node

    def get_child_node_with_id(self, query_id: str):
        for child_node in self.child_nodes:
            if child_node.id == query_id:
                return child_node
        return None

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
        actual_trips_required = max(self.node_family_actual_demand // self.vehicle_capacity, 1)
        remaining_capacity = self.node_family_actual_demand % self.vehicle_capacity

        if actual_trips_required == expected_trips_required:
            if remaining_capacity == 0 and actual_trips_required != 1:
                pass
            else:
                # Update demand of last node in child_nodes
                self.child_nodes[-1].expected_demand = remaining_capacity

        else:
            # Fewer / more trips are required than planned: self.child_nodes must shrink / grow
            # Clear existing child_nodes list
            self.child_nodes.clear()
            self.create_child_nodes(actual_trips_required, remaining_capacity)

    def __repr__(self):
        return f"Node family {self.node_family_id}, Child nodes: {self.child_nodes}"


