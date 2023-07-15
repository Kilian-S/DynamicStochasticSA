import copy
import random
import pandas as pd
import numpy as np
from numpy.random import rand
from errors.errors import *
from inputs.dynamic_distance_matrix import DynamicDistanceMatrix
from inputs.node import Node
from numpy import exp


def create_boolean_matrix(tours: list[list[any]]) -> pd.DataFrame:
    """
        Create a Boolean matrix representing the connections between nodes in the given SDCVRP

        Args:
            tours (list[list[any]]): A list of tours, where each tour is represented as a list of nodes.

        Returns:
            pd.DataFrame: A Boolean matrix where each element indicates whether there is a connection between two nodes.

    """
    # Get all unique nodes
    nodes = sorted(list(set(node for tour in tours for node in tour)))

    # Initialize the matrix with zeros
    matrix = pd.DataFrame(0, index=nodes, columns=nodes)

    # Populate the matrix
    for tour in tours:
        for i in range(len(tour) - 1):
            node1 = tour[i]
            node2 = tour[i + 1]
            matrix.at[node1, node2] = 1

    return matrix


def objective(tours: list[list[any]], distance_matrix: DynamicDistanceMatrix):
    """
        Calculate the objective value (total distance travelled) based on the given tours and distance matrix.

        Args:
            tours (list[list[any]]): A list of tours, where each tour is represented as a list of nodes.
            distance_matrix (DynamicDistanceMatrix): The DDM containing the distances between nodes.

        Returns:
            float: The calculated objective value.

    """
    x = create_boolean_matrix(tours)

    # Hadamard product
    objective_value = x.multiply(distance_matrix.matrix)

    return objective_value.values.sum()


def get_node_by_id(node_id: str, nodes: list[Node]) -> Node:
    """
        Get a node by its ID from a list of nodes.

        Args:
            node_id (str): The ID of the node to search for.
            nodes (list[Node]): The list of nodes to search in.

        Returns:
            Node: The node with the matching ID.

        Raises:
            NodeNotFoundError: If the node with the specified ID is not found in the list.

    """
    for node in nodes:
        if node.id == node_id:
            return node
    raise NodeNotFoundError


def is_single_family(nodes: list[Node]) -> bool:
    """
    Check if all nodes in the list belong to a single node family.

    Args:
        nodes (list[Node]): The list of nodes to check.

    Returns:
        bool: True if all nodes belong to a single family, False otherwise.

    """

    family_ids = set()
    for node in nodes:
        if node.id != '0':
            family_id = node.id.split('.')[0]
            family_ids.add(family_id)
            if len(family_ids) > 1:
                return False
    return True


def is_visitation(tours: list[list[any]], nodes: list[Node]) -> bool:
    """
    Check if all nodes (except for node 0) in the given list of nodes are visited in the tours.

    Args:
        tours (list[list[any]]): The list of tours.
        nodes (list[Node]): The list of nodes.

    Returns:
        bool: True if all nodes (except for node 0) are visited, False otherwise.

    """
    # Flatten the tours list and convert to a set for O(1) lookup time
    visited_nodes = set(node for tour in tours for node in tour)

    # Check that each node (except for node 0) appears in the visited set
    for node in nodes:
        if node.id not in visited_nodes and node.id != '0':
            return False  # Node was not visited

    return True  # All nodes were visited exactly once


def is_flow_conservation(tours: list[list[any]]) -> bool:
    """
    Check if the flow conservation condition is satisfied in the given tours.

    Args:
        tours (list[list[any]]): The list of tours.

    Returns:
        bool: True if the flow conservation condition is satisfied, False otherwise.

    """

    visit_counts = {node_id: 0 for tour in tours for node_id in tour}

    for tour in tours:
        for node_id in tour:
            visit_counts[node_id] += 1

    for node_id, count in visit_counts.items():
        if count != 1 and node_id != '0':
            return False

    return True


def is_within_vehicle_capacity(tours: list[list[any]], nodes: list[Node], vehicle_capacity: int) -> bool:
    """
    Check if all tours satisfy the vehicle capacity constraint.

    Args:
        tours (list[list[any]]): The list of tours.
        nodes (list[Node]): The list of nodes.
        vehicle_capacity (int): The vehicle capacity.

    Returns:
        bool: True if all tours satisfy the vehicle capacity constraint, False otherwise.

    """

    for tour in tours:
        tour_demand = sum(get_node_by_id(node_id, nodes).expected_demand for node_id in tour[1:-1])
        if tour_demand > vehicle_capacity and len(tour) > 3:
            return False

    return True


def is_correct_number_of_visits_set(nodes: list[Node], vehicle_capacity: int) -> bool:
    """
    Check if the expected demand of each node is within the vehicle capacity.

    Args:
        nodes (list[Node]): The list of nodes.
        vehicle_capacity (int): The vehicle capacity.

    Returns:
        bool: True if the expected demand of each node is within the vehicle capacity, False otherwise.

    """

    for node in nodes:
        if node.expected_demand > vehicle_capacity:
            return False

    return True


def starts_at_depot(tours: list[list[any]]) -> bool:
    """
    Check if each tour starts at the depot (node '0').

    Args:
        tours (list[list[any]]): The list of tours.

    Returns:
        bool: True if each tour starts at the depot, False otherwise.

    """

    for tour in tours:
        if tour[0] != '0':
            return False

    return True


def ends_at_depot(tours: list[list[any]]) -> bool:
    """
    Check if each tour ends at the depot (node '0').

    Args:
        tours (list[list[any]]): The list of tours.

    Returns:
        bool: True if each tour ends at the depot, False otherwise.

    """

    for tour in tours:
        if tour[-1] != '0':
            return False

    return True


def is_feasible(tours: list[list[any]], nodes: list[Node], vehicle_capacity: int) -> bool:
    """
        Check if a solution represented by a list of tours is feasible. Considers the following constraints: Visitation, flow conservation, vehicle capacity, correct number of
        visits, start at depot, end at depot.

        Args:
            tours (list[list[any]]): The list of tours.
            nodes (list[Node]): The list of nodes.
            vehicle_capacity (int): The capacity of the vehicles.

        Returns:
            bool: True if the solution is feasible, False otherwise.

    """

    return is_visitation(tours, nodes) and is_flow_conservation(tours) and is_within_vehicle_capacity(tours, nodes, vehicle_capacity) and \
           is_correct_number_of_visits_set(nodes, vehicle_capacity) and \
           starts_at_depot(tours) and ends_at_depot(tours)


def simulated_annealing(tours: list[list[any]], nodes: list[Node], distance_matrix: np.array, objective: callable, initial_temperature: int, iterations: int,
                        vehicle_capacity: int) -> tuple:
    """
    Apply the SA algorithm to optimise the SDCVRP. This method does not include dynamic constraints. As a result, it is used to determine the first SA solution in the iterative
    dynamic SA solution process.

    Args:
        tours (list[list[any]]): The initial tours.
        nodes (list[Node]): The list of nodes.
        distance_matrix (np.array): The distance matrix.
        objective (callable): The objective function.
        initial_temperature (int): The initial temperature for simulated annealing.
        iterations (int): The number of iterations for simulated annealing.
        vehicle_capacity (int): The vehicle capacity.

    Returns:
        tuple: A tuple containing the best objective function value and the optimised tours.

    Raises:
        InfeasibilityError: If the initial tours are infeasible.

    """

    # Check if input tours are feasible
    if not is_feasible(tours, nodes, vehicle_capacity):
        raise InfeasibilityError

    best_objective_function_value = objective(tours, distance_matrix)
    current_tours, current_tours_value = tours, best_objective_function_value

    # If there are fewer than three nodes, SA optimisation will not work.
    if len(nodes) <= 2 or is_single_family(nodes):
        return best_objective_function_value, tours

    i = 0

    while i < iterations:
        # Determine extraction index
        randomly_selected_extraction_tour_index = random.randrange(len(current_tours))
        # Randomly extract a node (NOT NODE INDEX!) from the randomly selected tour
        random_node = random.choice([node for node in current_tours[randomly_selected_extraction_tour_index] if node != '0'])

        candidate_tours = copy.deepcopy(current_tours)
        candidate_tours[randomly_selected_extraction_tour_index].remove(random_node)

        # Get rid of empty tours
        candidate_tours = [candidate_tour for candidate_tour in candidate_tours if len(candidate_tour) > 2]
        # Add an empty tour to the end of the list of tours
        candidate_tours.append([0, 0])

        # Determine insertion index
        randomly_selected_insertion_tour_index = random.randrange(len(candidate_tours))
        if len(candidate_tours[randomly_selected_insertion_tour_index]) > 2:
            insertion_index = random.randint(1, len(candidate_tours[randomly_selected_insertion_tour_index]) - 1)
        else:
            insertion_index = 1

        candidate_tours[randomly_selected_insertion_tour_index].insert(insertion_index, random_node)
        # Get rid of empty tours (ASSUMPTION: only the two depot nodes left in the removed tour)
        candidate_tours = [candidate_tour for candidate_tour in candidate_tours if len(candidate_tour) > 2]

        # Check if the newly created tour is feasible
        if is_feasible(candidate_tours, nodes, vehicle_capacity):
            candidate_tours_value = objective(candidate_tours, distance_matrix)

            if candidate_tours_value <= current_tours_value:
                # Update new best tour
                tours, best_objective_function_value = candidate_tours, candidate_tours_value
                print("Iteration: %d    Distance: %d    Tours: " % (i, candidate_tours_value), candidate_tours)

            # Possible acceptance of a worse solution based on Metropolis criterion
            difference = candidate_tours_value - current_tours_value
            t = initial_temperature / (1 + exp(0.1 * (i - 0.5 * iterations)))

            power = -difference / t
            exp_power_limit_lower = -20  # Below this, exp(x) is effectively 0
            exp_power_limit_upper = 700  # Above this, exp(x) causes overflow
            if power < exp_power_limit_lower:
                power = exp_power_limit_lower
            elif power > exp_power_limit_upper:
                power = exp_power_limit_upper
            metropolis = exp(power)

            if difference < 0 or rand() < metropolis:
                current_tours, current_tours_value = candidate_tours, candidate_tours_value

            i += 1
        else:
            continue

    print()
    return best_objective_function_value, tours


def is_traversal_subsequence_of_tours(tours: list[list[any]], traversal_states: list[list[any]]) -> bool:
    """
    Check if a given traversal sequence is a subsequence of the corresponding tours. A tour may be described as [0, 1, 2, 0]. Valid subsequences of this sequence are [0],
    [0, 1], [0, 1, 2], and [0, 1, 2, 0]. Although [1, 2, 0] and other similar variations are also subsequences, they are not valid because they do not begin and/or end at the
    depot.

    Args:
        tours (list[list[any]]): The list of tours.
        traversal_states (list[list[any]]): The list of traversal sequences.

    Returns:
        bool: True if the traversal sequence is a subsequence for each tour, False otherwise.

    """

    if len(tours) != len(traversal_states):
        return False

    for tour, traversal in zip(tours, traversal_states):
        if len(traversal) > len(tour):
            return False
        for i in range(len(traversal)):
            if tour[i] != traversal[i]:
                return False

    return True


def get_lock_indices(traversal_states: list[list[any]]) -> list[int]:
    """
    Determines for each tour the index of the node up to (and including) which that tour is locked. Essentially, once a node has been visited or traversed, no new nodes can be
    placed before nodes that have already been visited as this would constitute time travel.

    An index of 0 means that the first node is locked (as is always the case with the depot). An index of 1 means that the depot and the first node in the tour are locked.
    Locked nodes cannot be visited again, and no unvisited nodes can be placed before a locked node.

    :param traversal_states: current traversal state of each tour, tour 1 in tours corresponds to tour 1 in traversal_state
    :return: list[int]: a list of lock indices indicating up to which node index each tour is locked.
    """

    lock_indices = [len(traversal) - 1 for traversal in traversal_states]
    return lock_indices


def simulated_annealing_with_dynamic_constraints(tours: list[list[any]], nodes: list[Node], distance_matrix: np.array, objective: callable, initial_temperature: int,
                                                 iterations: int, vehicle_capacity: int, traversal_states: list[list[any]]) -> tuple:
    """
        Performs the SA algorithm on the SDCVRP. This function includes dynamic constriants (traversal states). It, therefore, is able to handle the progression of trucks
        through the SDCVRP.

        Parameters:
        tours (list[list[any]]): The initial tours to start the optimization from.
        nodes (list[Node]): List of nodes in the problem. Each node represents a city or a point to visit.
        distance_matrix (np.array): A 2D numpy array representing the distances between all pairs of nodes.
        objective (callable): Function to calculate the objective value for a set of tours.
        initial_temperature (int): The initial temperature for the Simulated Annealing algorithm.
        iterations (int): The number of iterations to run the Simulated Annealing algorithm.
        vehicle_capacity (int): The maximum capacity that each vehicle can carry.
        traversal_states (list[list[any]]): The current state of traversal for each tour.

        Returns:
        best_objective_function_value (float): The best (lowest) value of the objective function found during the search.
        tours (list[list[any]]): The set of tours associated with the best objective function value found.
        traversal_states (list[list[any]]): The state of traversal for each tour associated with the best objective function value found.

        Raises:
        InfeasibilityError: If the provided tours are not feasible given the nodes and vehicle capacity.
        IncorrectTraversalError: If the traversal states are not a subsequence of the tours.
    """

    if not is_feasible(tours, nodes, vehicle_capacity):
        raise InfeasibilityError

    if not is_traversal_subsequence_of_tours(tours, traversal_states):
        raise IncorrectTraversalError

    best_objective_function_value = objective(tours, distance_matrix)
    current_tours, current_tours_value = tours, best_objective_function_value

    current_traversal_states = traversal_states
    current_lock_indices = get_lock_indices(traversal_states)

    # TODO: Add check if the traversal states even allow for simulated annealing to occur
    if len(nodes) <= 2:
        return best_objective_function_value, tours

    i = 0

    while i < iterations:

        # Randomly select a tour extraction index. If the tour is completely locked, select a new index until this is not the case.
        while True:
            randomly_selected_extraction_tour_index = random.randrange(len(current_tours))
            if current_lock_indices[randomly_selected_extraction_tour_index] < len(current_tours[randomly_selected_extraction_tour_index]) - 2:
                break

        # Randomly select a node (NOT NODE INDEX!) from lock index onwards from the randomly selected tour
        random_node = random.choice([node for node in current_tours[randomly_selected_extraction_tour_index][current_lock_indices[randomly_selected_extraction_tour_index] + 1:]
                                     if node != '0'])
        candidate_tours = copy.deepcopy(current_tours)
        candidate_tours[randomly_selected_extraction_tour_index].remove(random_node)

        candidate_traversal_states = copy.deepcopy(current_traversal_states)

        # Get rid of empty tours from candidate_tours and candidate_traversal_states
        tmp = len(candidate_tours)
        candidate_tours = [candidate_tour for candidate_tour in candidate_tours if len(candidate_tour) > 2]

        if tmp - len(candidate_tours):
            del candidate_traversal_states[randomly_selected_extraction_tour_index]

        # Add an empty tour to the end of the list of tours
        candidate_tours.append(['0', '0'])
        candidate_traversal_states.append(['0'])

        # Update candidate_lock_indices
        candidate_lock_indices = get_lock_indices(candidate_traversal_states)

        # Randomly select an index of the tours for inserting the randomly selected node. Continue selecting until an uncompleted tour is found.
        while True:
            randomly_selected_insertion_tour_index = random.randrange(len(candidate_tours))
            if candidate_lock_indices[randomly_selected_insertion_tour_index] + 1 < len(candidate_tours[randomly_selected_insertion_tour_index]):
                break

        if len(candidate_tours[randomly_selected_insertion_tour_index]) > 2:
            insertion_index = random.randint(candidate_lock_indices[randomly_selected_insertion_tour_index] + 1, len(candidate_tours[randomly_selected_insertion_tour_index]) - 1)
        else:
            insertion_index = 1

        candidate_tours[randomly_selected_insertion_tour_index].insert(insertion_index, random_node)
        # Get rid of empty tours
        tmp = len(candidate_tours)
        candidate_tours = [candidate_tour for candidate_tour in candidate_tours if len(candidate_tour) > 2]

        if tmp - len(candidate_tours):
            del candidate_traversal_states[-1]

        candidate_lock_indices = get_lock_indices(candidate_traversal_states)

        # Check if the newly created tour is feasible
        if is_feasible(candidate_tours, nodes, vehicle_capacity):
            candidate_tours_value = objective(candidate_tours, distance_matrix)

            if candidate_tours_value <= current_tours_value:
                # Update new best tour
                tours, best_objective_function_value, traversal_states = candidate_tours, candidate_tours_value, candidate_traversal_states
                print(f"Iteration: {i}    Distance: {candidate_tours_value}    Tours: {candidate_tours}    Traversal states: {candidate_traversal_states}")

            # Possible acceptance of a worse solution based on Metropolis criterion
            difference = candidate_tours_value - current_tours_value
            t = initial_temperature / (1 + exp(0.1 * (i - 0.5 * iterations)))

            power = -difference / t
            exp_power_limit_lower = -20  # Below this, exp(x) is effectively 0
            exp_power_limit_upper = 700  # Above this, exp(x) causes overflow
            if power < exp_power_limit_lower:
                power = exp_power_limit_lower
            elif power > exp_power_limit_upper:
                power = exp_power_limit_upper
            metropolis = exp(power)

            if difference < 0 or rand() < metropolis:
                current_tours, current_tours_value, current_traversal_states, current_lock_indices = candidate_tours, candidate_tours_value, candidate_traversal_states, \
                                                                                                     candidate_lock_indices

            i += 1
        else:
            continue

    print()
    return best_objective_function_value, tours, traversal_states

# Exponential decay
# t = initial_temperature / float(i + 1)

# Linear
# t = initial_temperature - ((i * initial_temperature) / iterations)

# Concave
# a = (math.log(initial_temperature) / math.log(iterations))
# t = initial_temperature - pow(i, a)

# Sigmoid
# t = initial_temperature / (1 + exp(0.1 * (i - 0.5 * iterations)))
