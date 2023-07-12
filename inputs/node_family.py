from typing import List
from inputs.node import Node, InputNode


class NodeFamily:
    """A NodeFamily pragmatically represents a physical demand node. However, some nodes must be visited more than once. Every visitation to a physical node requires the
    creation of a child node. It follows, that the number of child nodes corresponds to: NumChildNodes = (ActualDemand % VehicleCapacity)+1. However, actual demand is only
    uncovered as the problem is solved. this causes the number of nodes in the problem instance to change dynamically."""
    def __init__(self, input_node: InputNode, vehicle_capacity: int, is_visited=False):
        """
        Initialize a NodeFamily object.

        Args:
            input_node (InputNode): The input node object.
            vehicle_capacity (int): The vehicle capacity for this node family.
            is_visited (bool, optional): Indicates whether the node family has been visited. Defaults to False.
        """
        self.node_family_id = input_node.id
        self.node_family_expected_demand = input_node.expected_demand
        self.node_family_actual_demand = input_node.actual_demand
        self.vehicle_capacity = vehicle_capacity
        self.child_nodes: List[Node] = []
        self.is_visited = is_visited

        self.initialise()

    def create_child_nodes(self, trips_required: int, remaining_capacity: int):
        """
        Create child nodes based on the given number of trips required and remaining capacity.

        :param trips_required: Number of trips required to fulfill the demand.
        :param remaining_capacity: Remaining capacity after the trips.
        """
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
        """
        Initialize the NodeFamily object by creating child nodes based on the expected demand and vehicle capacity.
        """
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
        """
        Update the node family by marking it as visited and updating the child nodes based on the actual demand.

        """
        self.is_visited = True

        if self.node_family_actual_demand == 0:
            self.child_nodes.clear()
            self.child_nodes.append(Node(f'{self.node_family_id}.{1}', self.node_family_actual_demand))

        else:
            actual_trips_required = self.node_family_actual_demand // self.vehicle_capacity
            remaining_capacity = self.node_family_actual_demand % self.vehicle_capacity

            self.child_nodes.clear()
            self.create_child_nodes(actual_trips_required, remaining_capacity)

    def __repr__(self):
        return f"Node family {self.node_family_id}, Child nodes: {self.child_nodes}"
