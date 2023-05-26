from inputs.node import Node


class NodeFamily:
    def __init__(self, node: Node, vehicle_capacity: int, is_visited=False):
        self.node_family_id = node.id
        self.node_family_expected_demand = node.expected_demand
        self.node_family_actual_demand = node.actual_demand
        self.child_nodes = []
        self.is_visited = is_visited

        self.initialise(vehicle_capacity, node)

    def initialise(self, vehicle_capacity, node):
        if self.node_family_expected_demand == 0:
            if self.node_family_id == 0:
                self.child_nodes = [Node('0', 0)]
            else:
                self.child_nodes = [Node(f'{self.node_family_id}.{1}', 0)]

        trips_required = self.node_family_expected_demand // vehicle_capacity
        remaining_capacity = self.node_family_expected_demand % vehicle_capacity

        for i in range(trips_required):
            new_node = Node(f'{self.node_family_id}.{i + 1}', vehicle_capacity)
            self.child_nodes.append(new_node)

        if remaining_capacity > 0:
            remaining_node = Node(f'{self.node_family_id}.{trips_required + 1}', remaining_capacity)
            self.child_nodes.append(remaining_node)

    def __repr__(self):
        return f"Node family {self.node_family_id}, Child nodes: {self.child_nodes}"


