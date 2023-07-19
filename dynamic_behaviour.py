import copy
import time
import numpy as np
from errors.errors import IncorrectReconciliationError, NodeNotFoundError
from inputs.dynamic_distance_matrix import DynamicDistanceMatrix, get_node_family_from_child_node
from inputs.dynamic_nodes_list import DynamicNodeList
from inputs.node import Node, InputNode
from inputs.node_family import NodeFamily
from simulated_annealing import objective, simulated_annealing, simulated_annealing_with_dynamic_constraints, is_within_vehicle_capacity


def remove_empty_tours(tours: list[list[str]]) -> tuple[list[list[str]], list[int]]:
    """
    Remove empty tours from the given list of tours.

    Args:
        tours (list[list[str]]): The list of tours.

    Returns:
        tuple[list[list[str]], list[int]]: A tuple containing the updated list of non-empty tours and a list of indices of the removed empty tours.

    """
    non_empty_tours = []
    removed_indices = []

    for i, tour in enumerate(tours):
        if len(tour) > 2:
            non_empty_tours.append(tour)
        else:
            removed_indices.append(i)

    return non_empty_tours, removed_indices


def create_initial_solution(nodes: list[Node]) -> list[list[any]]:
    """
    Create an initial solution consisting of tours that visit each node individually (single deliveries).

    Args:
        nodes (list[Node]): The list of nodes.

    Returns:
        list[list[any]]: The initial solution represented as a list of tours.

    """
    # Initialize an empty list to hold the tours
    tours = []

    # For each node (except depot), create a tour from depot to the node and back to the depot
    for node in nodes[1:]:
        tours.append(['0', node.id, '0'])

    return tours


def get_node_family(node_family_id: str, node_families: list[NodeFamily]) -> NodeFamily:
    """
    Retrieve the NodeFamily object with the specified node_family_id from the given list of NodeFamily objects.

    Args:
        node_family_id (str): The ID of the node family to retrieve.
        node_families (list[NodeFamily]): The list of NodeFamily objects.

    Returns:
        NodeFamily: The NodeFamily object with the specified node_family_id, or None if not found.

    """
    return next((node_family for node_family in node_families if node_family.node_family_id == node_family_id), None)


def remove_string_from_nested_list(nested_list: list[list[str]], string_to_remove: str) -> list[list[str]]:
    """
    Remove the specified string from all sublists within the given nested list.

    Args:
        nested_list (list[list[str]]): The nested list containing sublists of strings.
        string_to_remove (str): The string to remove from the sublists.

    Returns:
        list[list[str]]: The modified nested list with the specified string removed from the sublists.

    """
    for sublist in nested_list:
        if string_to_remove in sublist:
            sublist.remove(string_to_remove)
    return nested_list


def update_string_in_nested_list(nested_list: list[list[str]], old: str, new: str) -> list[list[str]]:
    """
    Update occurrences of a specified string with a new string within all sublists of the given nested list.

    Args:
        nested_list (list[list[str]]): The nested list containing sublists of strings.
        old (str): The string to be replaced.
        new (str): The new string to replace the old string.

    Returns:
        list[list[str]]: The modified nested list with the specified string updated to the new string.

    """
    for sublist in nested_list:
        for i in range(len(sublist)):
            if sublist[i] == old:
                sublist[i] = new
    return nested_list


def initialise_dynamic_data_structures(distance_matrix: np.array, nodes: list[InputNode], vehicle_capacity: int) -> tuple:
    """
    Initialise the dynamic data structures required for the dynamic programming approach.

    Args:
        distance_matrix (np.array): The distance matrix.
        nodes (list[Node]): The list of nodes.
        vehicle_capacity (int): The vehicle capacity.

    Returns:
        tuple: A tuple containing the initialised dynamic distance matrix, dynamic node list, node families, and nodes.

    """
    node_families = [NodeFamily(node, vehicle_capacity) for node in nodes]
    dynamic_distance_matrix = DynamicDistanceMatrix(distance_matrix, node_families)
    dynamic_node_list = DynamicNodeList(node_families, vehicle_capacity)
    nodes = dynamic_node_list.get_all_nodes()
    dynamic_distance_matrix.initialise(nodes)
    return dynamic_distance_matrix, dynamic_node_list, node_families, nodes


def initialise_current_variables(dynamic_distance_matrix, initial_temperature, iterations, nodes, objective, vehicle_capacity: int) -> tuple:
    """
    Initialise the current variables for the SA algorithm.

    Args:
        dynamic_distance_matrix: The dynamic distance matrix.
        initial_temperature: The initial temperature for the simulated annealing.
        iterations: The number of iterations for the simulated annealing.
        nodes: The list of nodes.
        objective: The objective function.
        vehicle_capacity: The vehicle capacity.

    Returns:
        tuple: A tuple containing the initial objective function value, current tours, and current traversal states.
    """
    current_tours = create_initial_solution(nodes)
    print(f'Initial Solu.   Distance: {objective(current_tours, dynamic_distance_matrix)}    Tours: {current_tours}\n')

    current_tours_value, current_tours = simulated_annealing(current_tours, nodes, dynamic_distance_matrix, objective, initial_temperature, iterations, vehicle_capacity)
    current_traversal_states = [['0'] for _ in current_tours]

    return current_tours_value, current_tours, current_traversal_states


def initialise_loop_visitation_variables(current_tours: list[list[str]], nodes: list[Node]) -> tuple:
    """
    Initialise the variables for loop visitation. These tours are essential for modelling the iterative progress of vehicles through the dynamic problem.

    Args:
        current_tours: The current tours.
        nodes: The list of nodes.

    Returns:
        tuple: A tuple containing the completed original tours, original tour positional index,
        original tours, and unvisited nodes.

    """
    unvisited_nodes = set(copy.deepcopy(nodes[1:]))
    original_tours = copy.deepcopy(current_tours)
    original_tour_positional_index = [0] * len(original_tours)
    completed_original_tours = []

    return completed_original_tours, original_tour_positional_index, original_tours, unvisited_nodes


def get_child_node_delta(visited_node_family: NodeFamily) -> tuple:
    """
    Get the change in the number of child nodes for a visited node family.

    Args:
        visited_node_family: The visited node family.

    Returns:
        tuple: A tuple containing the new number of child nodes and the old number of child nodes.

    """
    old_number_of_child_nodes = len(visited_node_family.child_nodes)
    visited_node_family.update()
    new_number_of_child_nodes = len(visited_node_family.child_nodes)
    return new_number_of_child_nodes, old_number_of_child_nodes


def get_ids_of_newly_added_child_nodes(tours: list[list[str]], current_nodes: list[Node]) -> list[str]:
    """
    Get the IDs of the newly added child nodes in the tours.

    Args:
        tours: A list of tours.
        current_nodes: A list of current nodes.

    Returns:
        list[str]: A list of IDs of the newly added child nodes.

    """
    # Flatten the tours list to a single list of node ids
    flattened_tour_ids = [node_id for sublist in tours for node_id in sublist]
    current_node_ids = [node.id for node in current_nodes]
    newly_added_ids = [node_id for node_id in current_node_ids if node_id not in flattened_tour_ids]

    return newly_added_ids


def get_tour_id_with_node_id(current_tours: list[list[str]], target_node_id: str) -> int:
    """
    Get the tour ID that contains the target node ID.

    Args:
        current_tours: A list of current tours.
        target_node_id: The target node ID.

    Returns:
        int: The tour ID that contains the target node ID, or None if not found.

    """
    return next((i for i, tour in enumerate(current_tours) if target_node_id in tour), None)


def get_tour_demand(tour: list[str], nodes: list[Node]):
    """
    Calculate the total demand of a tour.

    Args:
        tour: A list representing a tour.
        nodes: A list of Node objects.

    Returns:
        int: The total demand of the tour.

    """
    node_demand_dict = {node.id: node.expected_demand for node in nodes}
    total_demand = sum(node_demand_dict[node_id] for node_id in tour)
    return total_demand


def get_tour_demand_up_to_index(tour: list[str], nodes: list[Node], i: int):
    """
    Calculate the total demand of a tour up to a specified index.

    Args:
        tour: A list representing a tour.
        nodes: A list of Node objects.
        i: The index up to which the demand should be calculated.

    Returns:
        int: The total demand of the tour up to the specified index.

    """
    return get_tour_demand(tour[:i + 1], nodes)


def remove_indices_from_list(input_list: list, removed_indices: list[int]):
    """
    Remove elements from a list based on the specified indices.

    Args:
        input_list: The input list.
        removed_indices: A list of indices to be removed.

    Returns:
        list: A new list with the elements removed.

    """
    return [item for i, item in enumerate(input_list) if i not in removed_indices]


def get_shortfall(tour: list[str], nodes: list[Node], index: int, vehicle_capacity: int) -> int:
    """
    Calculate the shortfall (excess demand over vehicle capacity) at a specific index of a tour.

    Args:
        tour: A list of node IDs representing the tour.
        nodes: A list of Node objects representing all nodes.
        index: The index in the tour where the shortfall is calculated.
        vehicle_capacity: The capacity of the vehicle.

    Returns:
        int: The shortfall at the specified index.

    """
    return max(0, get_tour_demand_up_to_index(tour, nodes, index) - vehicle_capacity)


def get_child_node_index(tour: list[str], visited_node_family: NodeFamily) -> int:
    """
    Get the index of the first occurrence of a child node belonging to a specific visited node family in a tour.

    Args:
        tour: A list of node IDs representing the tour.
        visited_node_family: The visited NodeFamily object.

    Returns:
        int: The index of the child node in the tour.

    Raises:
        NodeNotFoundError: If the child node belonging to the visited node family is not found in the tour.

    """
    for i, node_id in enumerate(tour):
        try:
            node_family_id, _ = node_id.split('.', 1)
            if node_family_id == visited_node_family.node_family_id:
                return i

        except ValueError:
            # Skip the depot node (ID = '0'), which doesn't have a "."
            continue

    raise NodeNotFoundError


def reconcile_child_node_increase(dynamic_distance_matrix: DynamicDistanceMatrix, dynamic_node_list: DynamicNodeList, nodes: list[Node], unvisited_nodes: set[Node],
                                  visited_node_family: NodeFamily) -> list[Node]:
    """
    Reconcile the increase in child nodes of a visited node family with the dynamic distance matrix and unvisited nodes.

    Args:
        dynamic_distance_matrix: The DynamicDistanceMatrix object.
        dynamic_node_list: The DynamicNodeList object.
        nodes: The list of all nodes.
        unvisited_nodes: A set of unvisited nodes.
        visited_node_family: The visited NodeFamily object.

    Returns:
        list[Node]: The updated list of all nodes.

    """
    # Update unvisited_nodes
    updated_child_nodes = visited_node_family.child_nodes
    unaccounted_nodes = {node for node in updated_child_nodes if node not in nodes}
    unvisited_nodes.update(unaccounted_nodes)

    # Update dynamic_distance_matrix
    nodes = dynamic_node_list.get_all_nodes()
    dynamic_distance_matrix.update(nodes)

    return nodes


def reconcile_child_node_decrease(current_tours: list[list[str]], dynamic_distance_matrix: DynamicDistanceMatrix, dynamic_node_list: DynamicNodeList, next_node_in_tour: str,
                                  nodes: list[Node], original_tours: list[list[str]], unvisited_nodes: set[Node], visited_node_family: NodeFamily,
                                  current_traversal_states: list[list[str]], original_tour_positional_index: list[int]) -> tuple:
    """
        Reconcile the decrease in child nodes. This method is very complex because situations can arise where the ndoe that was just visited is deleted because of an
        overestimation of expected demand.

        Args:
            current_tours: The current tours.
            dynamic_distance_matrix: The DynamicDistanceMatrix object.
            dynamic_node_list: The DynamicNodeList object.
            next_node_in_tour: The next node in the tour.
            nodes: The list of all nodes.
            original_tours: The original tours.
            unvisited_nodes: A set of unvisited nodes.
            visited_node_family: The visited NodeFamily object.
            current_traversal_states: The current traversal states.
            original_tour_positional_index: The original tour positional indices.

        Returns:
            tuple: A tuple containing the updated current tours, next node in tour, nodes, original tours, unvisited nodes, current traversal states, and
                   original tour positional indices.

    """

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
    unvisited_nodes = {node for node in unvisited_nodes if node.id not in {deleted_node.id for deleted_node in deleted_nodes}}
    nodes = dynamic_node_list.get_all_nodes()
    dynamic_distance_matrix.update(nodes)
    return current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes, current_traversal_states, original_tour_positional_index


def end_tour_early(current_tours: list[list[str]], i: int, filled_index: int) -> tuple:
    """
    End a tour early by splitting it into a primary tour and a shortfall tour.

    Args:
        current_tours: The current tours.
        i: The index of the tour to end early.
        filled_index: The index indicating the last filled position in the tour.

    Returns:
        tuple: A tuple containing the primary tour and the shortfall tour.

    """
    primary_tour = current_tours[i][:filled_index + 1] + ['0']
    shortfall_tour = ['0'] + current_tours[i][filled_index + 1:]

    return primary_tour, shortfall_tour


def reconcile_tours_with_violated_capacity_constraint(current_tours: list[list[str]], original_tours: list[list[str]], unvisited_nodes: set[Node],
                                                      current_traversal_states: list[list[str]], nodes: list[Node], visited_node_family: NodeFamily, i: int,
                                                      dynamic_node_list: DynamicNodeList, vehicle_capacity: int) -> tuple:
    """
        Reconcile tours that violate the capacity constraint by adjusting the tours and creating new child nodes.

        Args:
            current_tours: The current tours.
            original_tours: The original tours.
            unvisited_nodes: The set of unvisited nodes.
            current_traversal_states: The current traversal states.
            nodes: The list of nodes.
            visited_node_family: The visited node family.
            i: The index of the tour.
            dynamic_node_list: The dynamic node list.
            vehicle_capacity: The vehicle capacity.

        Returns:
            tuple: A tuple containing the updated current tours, original tours, unvisited nodes, current traversal states, and nodes.

    """

    is_in_original_tours = current_tours[i] in original_tours

    child_node_index = get_child_node_index(current_tours[i], visited_node_family)
    shortfall = get_shortfall(current_tours[i], nodes, child_node_index, vehicle_capacity)

    if shortfall == 0:
        # End tour early (primary tour) and append the rest of the nodes (shortfall tour) to current_tours and (if the tour is in original_tours) to original_tours
        primary_tour, shortfall_tour = end_tour_early(current_tours, i, child_node_index)
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
        unvisited_nodes.add(shortfall_node)

        primary_tour, shortfall_tour = end_tour_early(current_tours, i, child_node_index)
        current_tours[i] = primary_tour

        # Decrease demand of the split node
        n = visited_node_family.get_child_node_with_id(current_tours[i][child_node_index])
        n.expected_demand -= shortfall

        if is_in_original_tours:
            original_tours[i] = primary_tour

        shortfall_tour.insert(1, shortfall_node.id)
        current_tours.append(shortfall_tour)
        current_traversal_states.append(['0'])

    nodes = dynamic_node_list.get_all_nodes()
    return current_tours, original_tours, unvisited_nodes, current_traversal_states, nodes


def check_vehicle_capacity_constraint_of_tours(current_tours: list[list[str]], original_tours: list[list[str]], visited_node_family: NodeFamily, unvisited_nodes: set[Node],
                                               current_traversal_states: list[list[str]], nodes: list[Node], dynamic_node_list: DynamicNodeList, vehicle_capacity: int) -> tuple:
    """
        Check the vehicle capacity constraint for each tour in the current tours and reconcile the tours if the constraint is violated.

        Args:
            current_tours: The current tours.
            original_tours: The original tours.
            visited_node_family: The visited node family.
            unvisited_nodes: The set of unvisited nodes.
            current_traversal_states: The current traversal states.
            nodes: The list of nodes.
            dynamic_node_list: The dynamic node list.
            vehicle_capacity: The vehicle capacity.

        Returns:
            tuple: A tuple containing the updated current tours, original tours, visited node family, unvisited nodes, and current traversal states.

    """

    # Check vehicle capacity constraint for each tour; If the constraint is violated reconciliation is necessary
    i = 0
    while i < len(current_tours):
        if not is_within_vehicle_capacity([current_tours[i]], nodes, vehicle_capacity):
            current_tours, original_tours, unvisited_nodes, current_traversal_states, nodes = reconcile_tours_with_violated_capacity_constraint(current_tours, original_tours,
                                                                                                                                                unvisited_nodes,
                                                                                                                                                current_traversal_states, nodes,
                                                                                                                                                visited_node_family, i,
                                                                                                                                                dynamic_node_list, vehicle_capacity)
        i += 1

    return current_tours, original_tours, visited_node_family, unvisited_nodes, current_traversal_states


def reconcile_expected_and_actual_demand(current_tours: list[list[str]], dynamic_distance_matrix: DynamicDistanceMatrix, dynamic_node_list: DynamicNodeList,
                                         next_node_in_tour: str, nodes: list[Node], original_tours: list[list[str]], unvisited_nodes: set[Node], visited_node_family: NodeFamily,
                                         current_traversal_states: list[list[str]], original_tour_positional_index: list[int], vehicle_capacity: int) -> tuple:
    """
    Reconcile the expected and actual demand of a visited node family and update the relevant variables.

    Args:
        current_tours: The current tours.
        dynamic_distance_matrix: The dynamic distance matrix.
        dynamic_node_list: The dynamic node list.
        next_node_in_tour: The next node in tour.
        nodes: The list of nodes.
        original_tours: The original tours.
        unvisited_nodes: The set of unvisited nodes.
        visited_node_family: The visited node family.
        current_traversal_states: The current traversal states.
        original_tour_positional_index: The original tour positional index.
        vehicle_capacity: The vehicle capacity.

    Returns:
        tuple: A tuple containing the updated current tours, next node in tour, nodes, original tours, unvisited nodes, current traversal states, and original tour positional index.

    """

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
        nodes = dynamic_node_list.get_all_nodes()
        current_tours, original_tours, visited_node_family, unvisited_nodes, current_traversal_states = check_vehicle_capacity_constraint_of_tours(current_tours, original_tours,
                                                                                                                                                   visited_node_family,
                                                                                                                                                   unvisited_nodes,
                                                                                                                                                   current_traversal_states, nodes,
                                                                                                                                                   dynamic_node_list,
                                                                                                                                                   vehicle_capacity)

    nodes = dynamic_node_list.get_all_nodes()
    return current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes, current_traversal_states, original_tour_positional_index


def update_loop_visitation_variables(completed_original_tours: list[list[str]], current_traversal_states: list[list[str]], i: int, next_node_in_tour: str,
                                     original_tours: list[list[str]], unvisited_nodes: set[Node]) -> set[Node]:
    """
    Update the variables related to loop visitation during the algorithm execution. Remember, these variables are essential for the dynamic "stepping" through the problem
    instance to work.

    Args:
        completed_original_tours: The list of completed original tours.
        current_traversal_states: The current traversal states.
        i: The index of the tour being updated.
        next_node_in_tour: The next node in the tour.
        original_tours: The original tours.
        unvisited_nodes: The set of unvisited nodes.

    Returns:
        set[Node]: The updated set of unvisited nodes.

    """

    current_traversal_states[i].append(next_node_in_tour)
    if next_node_in_tour == '0':
        # Tour has been completed
        completed_original_tours.append(original_tours[i])
    else:
        unvisited_nodes = {node for node in unvisited_nodes if node.id != next_node_in_tour}
    return unvisited_nodes


def integrate_newly_added_child_tours(tours: list[list[str]], nodes: list[Node], traversal_states: list[list[str]]) -> tuple:
    """
    Integrate the newly added child tours into the existing tours and traversal states.

    Args:
        tours: The list of tours.
        nodes: The list of nodes.
        traversal_states: The list of traversal states.

    Returns:
        tuple: The updated tours and traversal states.

    """

    newly_added_tours = get_ids_of_newly_added_child_nodes(tours, nodes)

    if len(newly_added_tours) == 0:
        return tours, traversal_states

    for node_id in newly_added_tours:
        tours.append(['0', node_id, '0'])
        traversal_states.append(['0'])

    return tours, traversal_states


def add_current_tour_to_original_tours(current_tours: list[list[str]], current_traversal_states: list[list[str]], original_tours: list[list[str]], tour: list[str]) -> tuple:
    """
    Add the current tour to the original tours list and corresponding traversal states.

    Args:
        current_tours: The list of current tours.
        current_traversal_states: The list of current traversal states.
        original_tours: The list of original tours.
        tour: The tour to be added to the original tours list.

    Returns:
        tuple: The updated current tours and current traversal states.

    """

    tour_index_in_current_tours = current_tours.index(tour)
    expected_index_in_original_tours = len(original_tours)

    if tour_index_in_current_tours != expected_index_in_original_tours:
        temp_tour = current_tours.pop(tour_index_in_current_tours)
        current_tours.insert(expected_index_in_original_tours, temp_tour)

        temp_traversal = current_traversal_states.pop(tour_index_in_current_tours)
        current_traversal_states.insert(expected_index_in_original_tours, temp_traversal)

    return current_tours, current_traversal_states


def update_original_tours(original_tours: list[list[str]], original_tour_positional_index: list[int], current_tours: list[list[str]], current_traversal_states: list[list[str]],
                          nodes: list[Node], vehicle_capacity: int, utilisation_target: float):
    """
    Update the original tours and their positional indices based on the current tours and traversal states.

    Args:
        original_tours: The list of original tours.
        original_tour_positional_index: The list of positional indices for the original tours.
        current_tours: The list of current tours.
        current_traversal_states: The list of current traversal states.
        nodes: The list of nodes.
        vehicle_capacity: The capacity of a vehicle.
        utilisation_target: The target utilisation level as a float.

    Returns:
        None

    """

    # Add tours to original_tours that are close to full
    for tour in current_tours:
        if tour not in original_tours and get_tour_demand(tour, nodes) / vehicle_capacity >= utilisation_target:
            current_tours, current_traversal_states = add_current_tour_to_original_tours(current_tours, current_traversal_states, original_tours, tour)

            original_tours.append(tour)
            original_tour_positional_index.append(0)

    # All current original_tours have been completed
    if set(tuple(tour) for tour in original_tours).issubset(tuple(state) for state in current_traversal_states):
        left_over_current_tours = [list(t) for t in set(tuple(tour) for tour in current_tours) - set(tuple(tour) for tour in original_tours)]
        for tour in left_over_current_tours:
            current_tours, current_traversal_states = add_current_tour_to_original_tours(current_tours, current_traversal_states, original_tours, tour)

            original_tours.append(tour)
            original_tour_positional_index.append(0)


def reconcile_new_and_current_sa_values(new_tours: list[list[str]], new_traversal_states: list[list[str]], original_tours: list[list[str]]) -> tuple:
    """
        Reconcile the order of tours and traversal states between the newly generated tours and the original tours.

        Args:
            new_tours: The list of newly generated tours.
            new_traversal_states: The list of newly generated traversal states.
            original_tours: The list of original tours.

        Returns:
            A tuple containing the reordered tours, reordered traversal states, and updated original tours.

    """

    reordered_tours = []
    reordered_traversal_states = []
    remaining_tours = new_tours.copy()
    remaining_traversal_states = new_traversal_states.copy()

    for original_tour in original_tours:
        index_in_remaining_tours = get_tour_id_with_node_id(remaining_tours, original_tour[1])

        reordered_tours.append(remaining_tours.pop(index_in_remaining_tours))
        reordered_traversal_states.append(remaining_traversal_states.pop(index_in_remaining_tours))

    reordered_tours.extend(remaining_tours)
    reordered_traversal_states.extend(remaining_traversal_states)

    for i in range(len(original_tours)):
        original_tours[i] = reordered_tours[i]

    return reordered_tours, reordered_traversal_states, original_tours


def dynamic_sa(nodes: list[InputNode], distance_matrix: np.array, objective: callable, initial_temperature: int, iterations: int, vehicle_capacity: int, utilisation_target: float):
    """
    This function performs a dynamic variant of Simulated Annealing (SA) to solve the Stochastic and Dynamic Capacitated Vehicle Routing Problem (SDCVRP) where the stochastic
    demand at each node is revealed dynamically.

    Args:
        nodes (list[InputNode]): A list of nodes in the problem, where each node is an instance of the InputNode class.
        distance_matrix (np.array): A 2D numpy array representing the distance between each pair of nodes.
        objective (callable): The objective function to be minimized in SA. This should be a function that takes a solution and returns a numerical score.
        initial_temperature (int): The starting temperature for the SA algorithm. High temperatures allow the algorithm to accept worse solutions to escape local minima.
        iterations (int): The number of iterations to perform in the SA algorithm.
        vehicle_capacity (int): The maximum capacity of each vehicle. This is a constraint in the VRP.
        utilisation_target (float): The target utilisation of each vehicle. This value should be between 0 and 1. The algorithm will attempt to ensure that the demand served by each vehicle is as close to this target as possible.

    Returns:
        current_tours_value: The total distance of the best solution found by the algorithm.
    """
    start_time = time.time()

    # Initialise node families, the dynamic node list that manages the node families, and the dynamic distance matrix
    dynamic_distance_matrix, dynamic_node_list, node_families, nodes = initialise_dynamic_data_structures(distance_matrix, nodes, vehicle_capacity)

    # Create initial solution
    current_tours_value, current_tours, current_traversal_states = initialise_current_variables(dynamic_distance_matrix, initial_temperature, iterations, nodes, objective,
                                                                                                vehicle_capacity)

    # All nodes are unvisited (we exclude the depot). We assume that there are at least as many trucks as there are routes in the first SA solution
    completed_original_tours, original_tour_positional_index, original_tours, unvisited_nodes = initialise_loop_visitation_variables(current_tours, nodes)

    while unvisited_nodes or original_tours != current_traversal_states:
        # Traverse each tour
        i = 0
        while i < len(original_tours):
            # Except if the tour has already ended
            if original_tour_positional_index[i] + 1 >= len(original_tours[i]) or original_tours[i] in completed_original_tours:
                i += 1
                continue

            next_node_in_tour = original_tours[i][original_tour_positional_index[i] + 1]

            # Check if node family has been visited before
            visited_node_family = get_node_family(get_node_family_from_child_node(next_node_in_tour), node_families)

            # Reconcile expected and actual demand
            current_tours, next_node_in_tour, nodes, original_tours, unvisited_nodes, current_traversal_states, original_tour_positional_index = reconcile_expected_and_actual_demand(
                current_tours, dynamic_distance_matrix, dynamic_node_list, next_node_in_tour, nodes, original_tours, unvisited_nodes, visited_node_family,
                current_traversal_states, original_tour_positional_index, vehicle_capacity)

            unvisited_nodes = update_loop_visitation_variables(completed_original_tours, current_traversal_states, i, next_node_in_tour, original_tours, unvisited_nodes)

            original_tour_positional_index[i] += 1
            i += 1

        # If new nodes were created, we need to add non-original tours to current_tours. These will also be optimised as part of simulated annealing, but not started until they
        # meet the vehicle capacity utilisation target.
        current_tours, current_traversal_states = integrate_newly_added_child_tours(current_tours, nodes, current_traversal_states)

        # Recalculate SA problem, if simulated annealing is possible (at least one unvisited node)
        if unvisited_nodes:
            dynamic_distance_matrix.update(nodes)
            current_tours_value, new_tours, new_traversal_states = simulated_annealing_with_dynamic_constraints(current_tours, nodes, dynamic_distance_matrix, objective,
                                                                                                                initial_temperature, iterations, vehicle_capacity,
                                                                                                                current_traversal_states)

            current_tours, current_traversal_states, original_tours = reconcile_new_and_current_sa_values(new_tours, new_traversal_states, original_tours)

            # Add non-original tours to original_tours; either if the expected tour demand is >= vehicle capacity, or if all other original tours are finished
            update_original_tours(original_tours, original_tour_positional_index, current_tours, current_traversal_states, nodes, vehicle_capacity, utilisation_target)

    end_time = time.time()
    execution_time = end_time - start_time
    nodes = dynamic_node_list.get_all_nodes()
    print(f'Final Solution.   Distance: {current_tours_value}    Tours: {current_tours}\n\n')
    print(f'Execution time: {execution_time}')
    return current_tours_value, current_tours, execution_time, nodes
