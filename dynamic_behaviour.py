import copy

from errors.errors import IncorrectReconciliationError
from global_parameters import *
from inputs.dynamic_distance_matrix import DynamicDistanceMatrix, get_node_family_from_child_node
from inputs.dynamic_nodes_list import DynamicNodeList
from inputs.node import Node
from inputs.node_family import NodeFamily
from simulated_annealing import objective, simulated_annealing, simulated_annealing_with_dynamic_constraints, is_within_vehicle_capacity


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


def remove_empty_tours(tours: list[list[str]]):
    non_empty_tours = []
    removed_indices = []

    for i, tour in enumerate(tours):
        if len(tour) > 2:
            non_empty_tours.append(tour)
        else:
            removed_indices.append(i)

    return non_empty_tours, removed_indices


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


def create_initial_solution(nodes: list[Node]):
    # Initialize an empty list to hold the tours
    tours = []

    # For each node (except depot), create a tour from depot to the node and back to the depot
    for node in nodes[1:]:
        tours.append(['0', node.id, '0'])

    return tours


def get_node_family(node_family_id: str, node_families: list[NodeFamily]):
    return next((node_family for node_family in node_families if node_family.node_family_id == node_family_id), None)


def remove_string_from_nested_list(nested_list: list[list[str]], string_to_remove: str):
    for sublist in nested_list:
        if string_to_remove in sublist:
            sublist.remove(string_to_remove)
    return nested_list


def update_string_in_nested_list(nested_list: list[list[str]], old: str, new: str):
    for sublist in nested_list:
        for i in range(len(sublist)):
            if sublist[i] == old:
                sublist[i] = new
    return nested_list


def initialise_dynamic_data_structures(distance_matrix, nodes, vehicle_capacity):
    node_families = [NodeFamily(node, vehicle_capacity) for node in nodes]
    dynamic_distance_matrix = DynamicDistanceMatrix(distance_matrix, node_families)
    dynamic_node_list = DynamicNodeList(node_families, vehicle_capacity)
    nodes = dynamic_node_list.get_all_nodes()
    dynamic_distance_matrix.initialise(nodes)
    return dynamic_distance_matrix, dynamic_node_list, node_families, nodes


def initialise_current_variables(dynamic_distance_matrix, initial_temperature, iterations, nodes, objective):
    current_tours = create_initial_solution(nodes)
    current_tours_value, current_tours = simulated_annealing(current_tours, nodes, dynamic_distance_matrix, objective, initial_temperature, iterations)
    current_traversal_states = [['0'] for _ in current_tours]
    return current_tours, current_traversal_states


def initialise_loop_visitation_variables(current_tours: list[list[str]], nodes: list[Node]):
    unvisited_nodes = set(copy.deepcopy(nodes[1:]))
    original_tours = copy.deepcopy(current_tours)
    original_tour_positional_index = [0] * len(original_tours)
    completed_original_tours = []
    return completed_original_tours, original_tour_positional_index, original_tours, unvisited_nodes


def get_child_node_delta(visited_node_family: NodeFamily):
    old_number_of_child_nodes = len(visited_node_family.child_nodes)
    visited_node_family.update()
    new_number_of_child_nodes = len(visited_node_family.child_nodes)
    return new_number_of_child_nodes, old_number_of_child_nodes


def get_ids_of_newly_added_child_nodes(tours: list[list[str]], current_nodes: list[Node]) -> list[str]:
    # Flatten the tours list to a single list of node ids
    flattened_tour_ids = [node_id for sublist in tours for node_id in sublist]
    current_node_ids = [node.id for node in current_nodes]
    newly_added_ids = [node_id for node_id in current_node_ids if node_id not in flattened_tour_ids]

    return newly_added_ids


def get_tour_id_with_node_id(current_tours: list[list[str]], target_node_id: str) -> int:
    return next((i for i, tour in enumerate(current_tours) if target_node_id in tour), None)


def get_tour_demand(tour: list[str], nodes: list[Node]):
    node_demand_dict = {node.id: node.expected_demand for node in nodes}
    total_demand = sum(node_demand_dict[node_id] for node_id in tour)
    return total_demand


def remove_indices_from_list(input_list: list, removed_indices: list[int]):
    return [item for i, item in enumerate(input_list) if i not in removed_indices]


def get_shortfall_info(tour: list[str], nodes: list[Node]):
    node_dict = {node.id: node for node in nodes}

    total_demand = 0
    filled_index = -1
    shortfall = 0

    for i, node_id in enumerate(tour):
        node = node_dict[node_id]
        total_demand += node.expected_demand

        if total_demand > VEHICLE_CAPACITY:
            filled_index = i
            shortfall = total_demand - VEHICLE_CAPACITY
            break
        elif total_demand == VEHICLE_CAPACITY:
            filled_index = i
            break

    return filled_index, shortfall


def reconcile_child_node_increase(dynamic_distance_matrix: DynamicDistanceMatrix, dynamic_node_list: DynamicNodeList, nodes: list[Node], unvisited_nodes: set[str],
                                  visited_node_family: NodeFamily):
    # Update unvisited_nodes
    updated_child_nodes = visited_node_family.child_nodes
    unaccounted_nodes = {node for node in updated_child_nodes if node not in nodes}
    unvisited_nodes.update(unaccounted_nodes)
    # Update dynamic_distance_matrix
    nodes = dynamic_node_list.get_all_nodes()
    dynamic_distance_matrix.update(nodes)
    return nodes


def reconcile_child_node_decrease(current_tours: list[list[str]], dynamic_distance_matrix: DynamicDistanceMatrix, dynamic_node_list: DynamicNodeList, next_node_in_tour: str,
                                  nodes: list[Node], original_tours: list[list[str]], unvisited_nodes: set[str], visited_node_family: NodeFamily,
                                  current_traversal_states: list[list[str]], original_tour_positional_index: list[int]):
    # Update unvisited_nodes
    updated_child_nodes = visited_node_family.child_nodes
    deleted_nodes = {node for node in nodes if node.id not in {child_node.id for child_node in updated_child_nodes} and node.id.split('.')[0] == visited_node_family.node_family_id
                     and node.id != '0'}

    if any(next_node_in_tour == node.id for node in deleted_nodes):
        # The node we have visited is now deleted; We need to set the node we just visited to one that exists. Since it is arbitrary which of the child_nodes of a
        # family node we visit first, we should always set the first-visited node to the first child node (indexed like 1.1 or 2.1) because this child node will
        # always exist
        # Update current_tours, next_node_in_tour, original_tours
        updated_next_node_in_tour = visited_node_family.node_family_id + ".1"

        # Remove the extant first child node from current tours and original tours
        current_tours = remove_string_from_nested_list(current_tours, updated_next_node_in_tour)
        original_tours = remove_string_from_nested_list(original_tours, updated_next_node_in_tour)

        # Update the tour I am currently on (replace the value of next_node_in_tour with updated_next_node_in_tour)
        current_tours = update_string_in_nested_list(current_tours, next_node_in_tour, updated_next_node_in_tour)
        original_tours = update_string_in_nested_list(original_tours, next_node_in_tour, updated_next_node_in_tour)
        next_node_in_tour = updated_next_node_in_tour

    # Remove the deleted child nodes from current_tours and original_tours
    for deleted_node in deleted_nodes:
        current_tours = remove_string_from_nested_list(current_tours, deleted_node.id)
        original_tours = remove_string_from_nested_list(original_tours, deleted_node.id)

    # Remove possible empty tours
    current_tours, removed_current_tour_indices = remove_empty_tours(current_tours)
    original_tours, removed_original_tour_indices = remove_empty_tours(original_tours)

    if removed_current_tour_indices != removed_original_tour_indices:
        raise IncorrectReconciliationError

    current_traversal_states = remove_indices_from_list(current_traversal_states, removed_current_tour_indices)
    original_tour_positional_index = remove_indices_from_list(original_tour_positional_index, removed_original_tour_indices)

    # Update unvisited_nodes, nodes, dynamic_distance_matrix
    unvisited_nodes = {node for node in unvisited_nodes if node not in {deleted_node.id for deleted_node in deleted_nodes}}
    nodes = dynamic_node_list.get_all_nodes()
    dynamic_distance_matrix.update(nodes)
    return current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes, current_traversal_states, original_tour_positional_index


def end_tour_early(current_tours: list[list[str]], i: int, filled_index: int):
    primary_tour = current_tours[i][:filled_index + 1] + ['0']
    shortfall_tour = ['0'] + current_tours[i][filled_index + 1:]

    return primary_tour, shortfall_tour


def reconcile_tours_with_violated_capacity_constraint(current_tours: list[list[str]], original_tours: list[list[str]], unvisited_nodes: set[str],
                                                      current_traversal_states: list[list[str]], nodes: list[Node], visited_node_family: NodeFamily, i: int,
                                                      dynamic_node_list: DynamicNodeList):
    is_in_original_tours = current_tours[i] in original_tours

    # Calculate the shortfall (unmet demand of the visited node)
    filled_index, shortfall = get_shortfall_info(current_tours[i], nodes)

    if shortfall == 0:
        # End tour early (primary tour) and append the rest of the nodes (shortfall tour) to current_tours and (if the tour is in original_tours) to original_tours
        primary_tour, shortfall_tour = end_tour_early(current_tours, i, filled_index)
        current_tours[i] = primary_tour

        if is_in_original_tours:
            original_tours[i] = primary_tour

        # Update current_tours with shortfall tour. The shortfall tour is not included as an original_tour
        if len(shortfall_tour) >= 3:
            current_tours.append(shortfall_tour)
            current_traversal_states.append(['0'])

    else:
        # Create a new child node that is part of the node family that embodies the shortfall
        shortfall_node = visited_node_family.create_new_child_node(shortfall)
        unvisited_nodes.add(shortfall_node.id)

        primary_tour, shortfall_tour = end_tour_early(current_tours, i, filled_index)
        current_tours[i] = primary_tour

        # Decrease demand of the split node
        n = visited_node_family.get_child_node_with_id(current_tours[i][filled_index])
        n.expected_demand -= shortfall

        if is_in_original_tours:
            original_tours[i] = primary_tour

        if len(shortfall_tour) >= 3:
            shortfall_tour.insert(1, shortfall_node.id)
            current_traversal_states.append(['0'])

    nodes = dynamic_node_list.get_all_nodes()
    return current_tours, original_tours, unvisited_nodes, current_traversal_states, nodes


def check_vehicle_capacity_constraint_of_tours(current_tours: list[list[str]], original_tours: list[list[str]], visited_node_family: NodeFamily, unvisited_nodes: set[str],
                                               current_traversal_states: list[list[str]], nodes: list[Node], dynamic_node_list: DynamicNodeList):
    # Check vehicle capacity constraint for each tour; If the constraint is violated reconciliation is necessary
    i = 0
    while i < len(current_tours):
        if not is_within_vehicle_capacity([current_tours[i]], nodes):
            current_tours, original_tours, unvisited_nodes, current_traversal_states, nodes = reconcile_tours_with_violated_capacity_constraint(current_tours, original_tours,
                                                                                                                                                unvisited_nodes,
                                                                                                                                                current_traversal_states, nodes,
                                                                                                                                                visited_node_family, i,
                                                                                                                                                dynamic_node_list)
        i += 1

    return current_tours, original_tours, visited_node_family, unvisited_nodes, current_traversal_states


def reconcile_expected_and_actual_demand(current_tours: list[list[str]], dynamic_distance_matrix: DynamicDistanceMatrix, dynamic_node_list: DynamicNodeList,
                                         next_node_in_tour: str, nodes: list[Node], original_tours: list[list[str]], unvisited_nodes: set[str], visited_node_family: NodeFamily,
                                         current_traversal_states: list[list[str]], original_tour_positional_index: list[int]):
    if not visited_node_family.is_visited:
        # Node family has not been visited: Check if recalculation of children nodes based on actual demand instead of expected demand is necessary
        new_number_of_child_nodes, old_number_of_child_nodes = get_child_node_delta(visited_node_family)

        # Check if child_nodes of node family has grown / shrunk
        if new_number_of_child_nodes == old_number_of_child_nodes:
            pass

        elif new_number_of_child_nodes > old_number_of_child_nodes:
            # Number of visits has increased
            nodes = reconcile_child_node_increase(dynamic_distance_matrix, dynamic_node_list, nodes, unvisited_nodes, visited_node_family)

        else:
            # Number of visits has decreased
            current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes, current_traversal_states, original_tour_positional_index = reconcile_child_node_decrease(
                current_tours, dynamic_distance_matrix, dynamic_node_list, next_node_in_tour, nodes, original_tours, unvisited_nodes, visited_node_family, current_traversal_states,
                original_tour_positional_index)

        # Based on the constellation of expected and actual demand, those tours that include the first and/or last node of the visited node family may have become infeasible.
        # The capacity constraint must be checked against the tours that include the first and/or last child nodes of the visited node family
        current_tours, original_tours, visited_node_family, unvisited_nodes, current_traversal_states = check_vehicle_capacity_constraint_of_tours(current_tours, original_tours,
                                                                                                                                                   visited_node_family,
                                                                                                                                                   unvisited_nodes,
                                                                                                                                                   current_traversal_states, nodes,
                                                                                                                                                   dynamic_node_list)

    return current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes, current_traversal_states, original_tour_positional_index


def update_loop_visitation_variables(completed_original_tours: list[list[str]], current_traversal_states: list[list[str]], i: int, next_node_in_tour: str,
                                     original_tours: list[list[str]], unvisited_nodes: set[str]):
    current_traversal_states[i].append(next_node_in_tour)
    if next_node_in_tour == 0:
        # Tour has been completed
        completed_original_tours.append(original_tours[i])
    else:
        unvisited_nodes = {node for node in unvisited_nodes if node != next_node_in_tour}
    return unvisited_nodes


def integrate_newly_added_child_tours(tours: list[list[str]], nodes: list[Node], traversal_states: list[list[str]]):
    newly_added_tours = get_ids_of_newly_added_child_nodes(tours, nodes)

    if len(newly_added_tours) == 0:
        return tours, traversal_states

    for node_id in newly_added_tours:
        tours.append(['0', node_id, '0'])
        traversal_states.append(['0'])

    return tours, traversal_states


def dynamic_sa(nodes: list[InputNode], distance_matrix: np.array, objective: callable, initial_temperature: int, iterations: int, vehicle_capacity: int):
    # Initialise node families, the dynamic node list that manages the node families, and the dynamic distance matrix
    dynamic_distance_matrix, dynamic_node_list, node_families, nodes = initialise_dynamic_data_structures(distance_matrix, nodes, vehicle_capacity)

    # Create initial solution
    current_tours, current_traversal_states = initialise_current_variables(dynamic_distance_matrix, initial_temperature, iterations, nodes, objective)

    # All nodes are unvisited (we exclude the depot). We assume that there are at least as many trucks as there are routes in the first SA solution
    completed_original_tours, original_tour_positional_index, original_tours, unvisited_nodes = initialise_loop_visitation_variables(current_tours, nodes)

    while unvisited_nodes or original_tours != current_traversal_states:
        # Traverse each tour
        i = 0
        while i < len(original_tours):
            # Except if the tour has already ended
            if original_tour_positional_index[i] + 1 >= len(original_tours[i]) or original_tours[i] in completed_original_tours:
                continue

            next_node_in_tour = original_tours[i][original_tour_positional_index[i] + 1]

            # Check if node family has been visited before
            visited_node_family = get_node_family(get_node_family_from_child_node(next_node_in_tour), node_families)

            current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes, current_traversal_states, original_tour_positional_index = reconcile_expected_and_actual_demand(
                current_tours, dynamic_distance_matrix, dynamic_node_list, next_node_in_tour, nodes, original_tours, unvisited_nodes, visited_node_family,
                current_traversal_states, original_tour_positional_index)

            unvisited_nodes = update_loop_visitation_variables(completed_original_tours, current_traversal_states, i, next_node_in_tour, original_tours, unvisited_nodes)

            original_tour_positional_index[i] += 1
            i += 1

        # Recalculate SA problem, if simulated annealing is possible (at least one unvisited node)
        if unvisited_nodes:
            # If new nodes were created, we need to add non-original tours to current_tours. These will also be optimised as part of simulated annealing, but not started until
            current_tours, current_traversal_states = integrate_newly_added_child_tours(current_tours, nodes, current_traversal_states)

            new_tours_value, new_tours, new_traversal_states = simulated_annealing_with_dynamic_constraints(current_tours, nodes, dynamic_distance_matrix, objective,
                                                                                                            initial_temperature, iterations, current_traversal_states)


dynamic_sa(NODES, SYM_DISTANCE_MATRIX, objective, INITIAL_TEMP, ITERATIONS, VEHICLE_CAPACITY)
