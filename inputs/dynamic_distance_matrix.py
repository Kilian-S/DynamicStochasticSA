import numpy as np
import pandas as pd


class DynamicDistanceMatrix:
    def __init__(self, distance_matrix: np.array, nodes: list[int]):
        assert len(nodes) == distance_matrix.shape[0] == distance_matrix.shape[1], "Nodes list size must be equal to the dimensions of the distance matrix"
        assert (distance_matrix.T == distance_matrix).all(), "Distance matrix must be symmetrical"

        nodes_as_str = [str(node) for node in nodes]
        self.matrix = pd.DataFrame(distance_matrix, index=nodes_as_str, columns=nodes_as_str)

    def split_node(self, node: str, split_into: int):
        assert split_into >= 2, "Number of resulting nodes should be at least 2"
        assert node in self.matrix.index, "Node not found in the matrix"

        # Create new nodes
        new_nodes = [f'{node}.{i}' for i in range(1, split_into)]

        # Copy data from original node to new nodes
        for new_node in new_nodes:
            self.matrix[new_node] = self.matrix[node]
            self.matrix.loc[new_node] = self.matrix.loc[node]

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


# distance_matrix = np.array([[0, 4, 7], [4, 0, 8], [7, 8, 0]])
# nodes = [1, 2, 3]
# ddm = DynamicDistanceMatrix(distance_matrix, nodes)
# print(ddm.matrix)
#
# ddm.split_node('1', 3)
# print(ddm.matrix)
#
# ddm.combine_nodes(['1', '1.1', '1.2'])
# print(ddm.matrix)
