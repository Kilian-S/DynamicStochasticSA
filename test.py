from global_parameters import *
from nodes import Node


def split_large_nodes(distance_matrix, nodes, max_capacity):
    new_nodes = []
    node_mapping = []  # Maps old node ids to new node ids
    next_id = 0

    # Iterate over nodes and split large ones
    for node in nodes:
        if node.demand > max_capacity:
            n_splits = node.demand // max_capacity
            remain_demand = node.demand % max_capacity

            for _ in range(n_splits):
                new_nodes.append(Node(next_id, max_capacity))
                node_mapping.append(node.id)
                next_id += 1

            if remain_demand > 0:
                new_nodes.append(Node(next_id, remain_demand))
                node_mapping.append(node.id)
                next_id += 1
        else:
            new_nodes.append(Node(next_id, node.demand))
            node_mapping.append(node.id)
            next_id += 1

    # Create new distance matrix
    size = len(new_nodes)
    new_distance_matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            old_i = node_mapping[i]
            old_j = node_mapping[j]
            new_distance_matrix[i][j] = distance_matrix[old_i][old_j]

    return new_distance_matrix, new_nodes


print(split_large_nodes(DISTANCE_MATRIX, NODES, VEHICLE_CAPACITY))






