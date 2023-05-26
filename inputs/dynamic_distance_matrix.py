from collections import defaultdict
from typing import List, Dict
import numpy as np
import pandas as pd
from inputs.node import Node
from inputs.node_family import NodeFamily


def group_by_family(nodes: List[Node]) -> Dict[str, List[Node]]:
    node_dict = defaultdict(list)

    for node in nodes:
        # Split the node id to get the family id
        node_family_id = node.id.split('.')[0]
        node_dict[node_family_id].append(node)

    return node_dict


class DynamicDistanceMatrix:
    def __init__(self, numpy_distance_matrix: np.array, node_families: list[NodeFamily]):
        assert len(node_families) == numpy_distance_matrix.shape[0] == numpy_distance_matrix.shape[1], "Node family list size must be equal to the dimensions of the distance matrix"
        assert (numpy_distance_matrix.T == numpy_distance_matrix).all(), "Distance matrix must be symmetrical"

        self.node_families = node_families

        # Convert node family ids to strings
        node_families_as_str = [str(node_family.node_family_id) for node_family in node_families]
        self.matrix = pd.DataFrame(numpy_distance_matrix, index=node_families_as_str, columns=node_families_as_str)

    def __str__(self):
        return str(self.matrix)

    def initialise(self, nodes: list[Node]):
        """
        This function assumes that the distance matrix is uninitialised. this means that the number of rows and columns has been determined by the number of node families. Now,
        these labels will be replaced by those of the node families' children (1.1 instead of 1). Moreover, if an expansion of the distance matrix is necessary (when there are
        more than one child nodes in a node family), the matrix will be expanded accordingly
        :param nodes: Input nodes. All should be indexed "NODE_FAMILY.NODE"
        :return: Updated dynamic distance matrix
        """

        input_node_families = set(node.id.split(".")[0] for node in nodes)
        matrix_node_families = set(str(node_family.node_family_id) for node_family in self.node_families)

        if input_node_families != matrix_node_families:
            raise ValueError("The node families in input nodes do not match those in the distance matrix.")

        nodes_grouped_by_family = group_by_family(nodes)

        for matrix_node_family in matrix_node_families:
            self.split_node(matrix_node_family, len(nodes_grouped_by_family[matrix_node_family]))

    def update(self, nodes: list[Node]):
        """

        :param nodes: The current nodes that exist in the VRP problem. These can be nodes proper, or dummy nodes used to represent a second visit to a node
        :return: Dynamic distance matrix with the correct dimensions.
        """
        # Find number of node families in the input nodes
        input_node_family_grouping = group_by_family(nodes)

        if len(input_node_family_grouping) != len(self.node_families):
            raise ValueError("Number of nodes does not match the size of the distance matrix.")

    def split_node(self, node: str, split_into: int):
        """

        :param node:
        :param split_into:
        :return:
        """
        assert node in self.matrix.index, "Node not found in the matrix"

        if node == '0':
            return self.matrix

        # Create new nodes
        new_nodes = [f'{node}.{i + 1}' for i in range(split_into)]

        # Copy data from original node to new nodes
        for new_node in new_nodes:
            self.matrix[new_node] = self.matrix[node]
            self.matrix.loc[new_node] = self.matrix.loc[node]

        # Delete the original node
        self.matrix.drop([node], axis=1, inplace=True)
        self.matrix.drop([node], axis=0, inplace=True)

        # Make sure the matrix is symmetrical
        self.matrix = self.matrix.T.sort_index().T.sort_index()

    def combine_nodes(self, nodes: list[str]):
        assert len(nodes) >= 2, "At least two nodes must be provided for combination"

        # Check that all nodes exist in the matrix
        for node in nodes:
            assert node in self.matrix.index, f"Node {node} not found in the matrix"

        # Check that all nodes belong to the same family
        node_families = {node.split('.')[0] for node in nodes}
        assert len(node_families) == 1, "All nodes must belong to the same family"

        # Find the node with the smallest name (smallest decimal part)
        min_node = min(nodes, key=lambda node: float('.'.join(node.split('.')[1:] or ['0'])))

        # Combine the nodes
        for node in nodes:
            if node != min_node:
                self.matrix[min_node] = self.matrix[min_node].combine(self.matrix[node], max)
                self.matrix.loc[min_node] = self.matrix.loc[min_node].combine(self.matrix.loc[node], max)
                self.matrix.drop(columns=node, inplace=True)
                self.matrix.drop(index=node, inplace=True)


# distance_matrix = np.array([[0, 4, 7, 5], [4, 0, 8, 3], [7, 8, 0, 6], [5, 3, 6, 0]])
# vehicle_capacity = 100  # Assuming a vehicle capacity of 100 units
#
# # Creating Node objects for depot and locations
# depot = Node(0, 0)  # Assuming id=0 and demand=0 for depot
# location1 = Node(1, 150)  # Assuming id=1 and demand=150 for location1
# location2 = Node(2, 200)  # Assuming id=2 and demand=200 for location2
# location3 = Node(3, 250)  # Assuming id=3 and demand=250 for location3
#
# # Creating NodeFamily objects
# depot_family = NodeFamily(depot, vehicle_capacity, is_visited=True)  # Depot is typically visited initially
# location1_family = NodeFamily(location1, vehicle_capacity)
# location2_family = NodeFamily(location2, vehicle_capacity)
# location3_family = NodeFamily(location3, vehicle_capacity)
#
# # Creating a list of NodeFamily objects
# node_families = [depot_family, location1_family, location2_family, location3_family]
# nodes = [Node("0", 0), Node("1.1", 100), Node("1.2", 50), Node("2.1", 100), Node("2.2", 100), Node("3.1", 100), Node("3.2", 100), Node("3.3", 50)]
# ddm = DynamicDistanceMatrix(distance_matrix, node_families)
# print(ddm)
# ddm.initialise(nodes)
# print(ddm)


