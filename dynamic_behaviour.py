import copy
from global_parameters import *
from inputs.dynamic_distance_matrix import DynamicDistanceMatrix
from inputs.dynamic_nodes_list import DynamicNodeList
from inputs.node import Node
from inputs.node_family import NodeFamily
from simulated_annealing import objective, simulated_annealing, simulated_annealing_with_dynamic_constraints


def calculate_required_tours(nodes: list[Node], vehicle_capacity: int):
    required_tours_dict = {}
    for node in nodes:
        # If the node is a depot node
        if node.id == 0:
            required_tours_dict[node.id] = 0
        # If the node is a non-depot node with zero demand
        elif node.expected_demand == 0:
            required_tours_dict[node.id] = 1
        # If the node is a non-depot node with demand between 1 and vehicle capacity
        elif 0 < node.expected_demand <= vehicle_capacity:
            required_tours_dict[node.id] = 1
        # If the node has a demand that exceeds the vehicle capacity
        else:
            required_tours_dict[node.id] = (node.expected_demand // vehicle_capacity) + (1 if node.expected_demand % vehicle_capacity else 0)

    return required_tours_dict


def expand_distance_matrix(nodes_dict: dict, distance_matrix: np.array):
    # Create a list to store the expanded IDs
    expanded_ids = []
    for node, required_stops in nodes_dict.items():
        # Repeat the node ID as many times as visits are required
        expanded_ids.extend([node] * max(1, required_stops))

    # Create a new distance matrix with expanded size
    new_matrix_size = len(expanded_ids)
    new_distance_matrix = np.zeros((new_matrix_size, new_matrix_size))

    # Fill the new distance matrix
    for i in range(new_matrix_size):
        for j in range(new_matrix_size):
            # Map the expanded IDs back to the original IDs
            original_i = expanded_ids[i]
            original_j = expanded_ids[j]
            # Assign the corresponding distance from the original distance matrix
            new_distance_matrix[i][j] = distance_matrix[original_i][original_j]

    return new_distance_matrix


def expand_nodes(nodes: list[Node], vehicle_capacity: int):
    pass


def create_initial_solution(nodes: list[Node]):
    # Initialize an empty list to hold the tours
    tours = []

    # For each node (except depot), create a tour from depot to the node and back to the depot
    for node in nodes[1:]:
        tours.append([0, node.id, 0])

    return tours


def dynamic_sa(nodes: list[Node], distance_matrix: np.array, objective: callable, initial_temperature: int, iterations: int, vehicle_capacity):
    # Initialise node families, the dynamic node list that manages the node families, and the
    node_families = [NodeFamily(node, vehicle_capacity) for node in nodes]
    dynamic_distance_matrix = DynamicDistanceMatrix(distance_matrix, node_families)
    dynamic_node_list = DynamicNodeList(node_families, vehicle_capacity)

    # Get all nodes from all node families
    nodes = dynamic_node_list.get_all_nodes()

    # Amend distance matrix
    dynamic_distance_matrix.initialise(nodes)

    # Create initial solution
    current_tours = create_initial_solution(nodes)
    current_tours_value, current_tours = simulated_annealing(current_tours, nodes, distance_matrix, objective, initial_temperature, iterations)
    current_traversal_states = [[0] for _ in current_tours]

    # All nodes are unvisited (we exclude the depot). We assume that there are at least as many trucks as there are routes in the first SA solution
    unvisited_nodes = copy.deepcopy(nodes[1:])
    original_tours = copy.deepcopy(current_tours)
    original_tour_positional_index = [0] * len(original_tours)
    completed_original_tours = []

    while unvisited_nodes or original_tours != current_traversal_states:
        # Traverse each tour
        for i in range(0, len(original_tours)):
            # By one step
            # Except if the tour has already ended
            if original_tour_positional_index[i] >= len(original_tours[i]) or original_tours[i] in completed_original_tours:
                continue

            next_node_in_tour = original_tours[i][original_tour_positional_index[i]+1]

            # Tour has been completed
            current_traversal_states[i].append(next_node_in_tour)
            if next_node_in_tour == 0:
                completed_original_tours.append(original_tours[i])
            else:
                unvisited_nodes = [node for node in unvisited_nodes if node.id != next_node_in_tour]

            original_tour_positional_index[i] += 1

        # Recalculate SA problem, if simulated annealing is possible (at least one unvisited node)
        if unvisited_nodes:
            new_tours_value, new_tours, new_traversal_states = simulated_annealing_with_dynamic_constraints(current_tours, nodes, distance_matrix, objective,
                                                                                                            initial_temperature, iterations, current_traversal_states)











dynamic_sa(NODES, SYM_DISTANCE_MATRIX, objective, INITIAL_TEMP, ITERATIONS, VEHICLE_CAPACITY)
























