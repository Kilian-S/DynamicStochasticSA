import copy
import random
from numpy.random import rand
from errors import *
from global_parameters import *
from nodes import Node
from numpy import exp


def create_boolean_matrix(tours: list[list[any]]):
    n = max(max(tour) for tour in tours) + 1
    matrix = np.zeros((n, n), dtype=int)

    for tour in tours:
        for i in range(len(tour) - 1):
            node1 = tour[i]
            node2 = tour[i + 1]
            matrix[node1][node2] = 1

    return matrix


def objective(tours: list[list[any]], distance_matrix: np.array):
    x = create_boolean_matrix(tours)
    objective_value = x * distance_matrix

    return np.sum(objective_value)


def is_visitation(tours: list[list[any]], nodes: list[Node]):
    # Flatten the tours list and convert to a set for O(1) lookup time
    visited_nodes = set(node for tour in tours for node in tour)

    # Check that each node (except for node 0) appears in the visited set
    for node in nodes:
        if node.id not in visited_nodes and node.id != 0:
            return False  # Node was not visited

    return True  # All nodes were visited exactly once


def is_flow_conservation(tours: list[list[any]]) -> bool:
    # Create a dictionary to count how many times each node is visited
    visit_counts = {node_id: 0 for tour in tours for node_id in tour}

    for tour in tours:
        # Increment visit counts for nodes in the tour
        for node_id in tour:
            visit_counts[node_id] += 1

    # # Check that all nodes except for the depot are visited exactly once
    # for node_id, count in visit_counts.items():
    #     if count != 1 and node_id != 0:
    #         return False

    return True  # Flow conservation condition satisfied


def is_within_vehicle_capacity(tours: list[list[any]], nodes: list[Node]):
    for tour in tours:
        tour_demand = sum(nodes[node].demand for node in tour)
        if tour_demand > VEHICLE_CAPACITY and len(tour) > 3:
            return False
    return True


def is_correct_number_of_visits_set(nodes: list[Node]):
    for node in nodes:
        if node.demand > VEHICLE_CAPACITY:
            return False
    return True


def starts_at_depot(tours: list[list[any]]):
    for tour in tours:
        if tour[0] != 0:
            return False
    return True


def ends_at_depot(tours: list[list[any]]):
    for tour in tours:
        if tour[-1] != 0:
            return False
    return True


def is_feasible(tours: list[list[any]], nodes: list[Node]):
    return is_visitation(tours, nodes) and is_flow_conservation(tours) and is_within_vehicle_capacity(tours, nodes) and is_correct_number_of_visits_set(nodes) and starts_at_depot(
        tours) and ends_at_depot(tours)


def simulated_annealing(tours: list[list[any]], nodes: list[Node], distance_matrix: np.array, objective: callable, initial_temperature: int, iterations: int):
    # Check if input tour is feasible
    if not is_feasible(tours, nodes):
        raise InfeasibilityError

    best_objective_function_value = objective(tours, distance_matrix)
    current_tours, current_tours_value = tours, best_objective_function_value
    i = 0

    while i < iterations:
        # Randomly select an index of the tours
        # TODO: should random extraction be dependent on number of tours or on number of nodes of a tour (normalise probability by number of non-depot nodes)? insertion?
        randomly_selected_extraction_tour_index = random.randrange(len(current_tours))
        # Randomly select a node (NOT NODE INDEX!) from the randomly selected tour
        random_node = random.choice([node for node in current_tours[randomly_selected_extraction_tour_index] if node != 0])

        candidate_tours = copy.deepcopy(current_tours)
        candidate_tours[randomly_selected_extraction_tour_index].remove(random_node)
        # Get rid of empty tours (ASSUMPTION: only the two depot nodes left in the removed tour)
        candidate_tours = [candidate_tour for candidate_tour in candidate_tours if len(candidate_tour) > 2]

        # Add an empty tour
        candidate_tours.append([0, 0])

        randomly_selected_insertion_tour_index = random.randrange(len(candidate_tours))
        if len(candidate_tours[randomly_selected_insertion_tour_index]) > 2:
            insertion_index = random.randint(1, len(candidate_tours[randomly_selected_insertion_tour_index]) - 1)
        else:
            insertion_index = 1

        candidate_tours[randomly_selected_insertion_tour_index].insert(insertion_index, random_node)
        # Get rid of empty tours (ASSUMPTION: only the two depot nodes left in the removed tour)
        candidate_tours = [candidate_tour for candidate_tour in candidate_tours if len(candidate_tour) > 2]

        if is_feasible(candidate_tours, nodes):
            candidate_tours_value = objective(candidate_tours, distance_matrix)

            # Should this be a LESS or LESSEQUAL?
            if candidate_tours_value <= current_tours_value:
                # Update new best tour
                tours, best_objective_function_value = candidate_tours, candidate_tours_value
                print("Iteration: %d    Distance: %d    Tours: " % (i, candidate_tours_value), candidate_tours)

                # Possible acceptance based on Metropolis criterion
                difference = candidate_tours_value - current_tours_value

                t = initial_temperature / float(i + 1)

                metropolis = exp(-difference / t)

                if difference < 0 or rand() < metropolis:
                    current_tours, current_tours_value = candidate_tours, candidate_tours_value

                i += 1
        else:
            continue

    # TODO: Return type of best_objective_function_value is a numpy float
    return best_objective_function_value, tours


def is_traversal_ordered_subset_of_tours(tours: list[list[any]], traversal_state: list[list[any]]):
    if len(tours) != len(traversal_state):
        return False

    for tour, traversal in zip(tours, traversal_state):
        if len(traversal) > len(tour):
            return False
        for i in range(len(traversal)):
            if tour[i] != traversal[i]:
                return False
    return True


def determine_lock_indices(tours: list[list[any]], traversal_state: list[list[any]]):
    """
    Determines for each tour the index of the node up to (and including) which that tour is locked.

    An index of 0 means that the first node is locked (as is always the case with the depot). An index of 1 means that the depot and the the first node in the tour are locked.
    Locked nodes cannot be visited again, and no nodes can be placed between two locked nodes.

    :param tours: Complete and feasible list of tours
    :param traversal_state: current traversal state of each tour, tour 1 in tours corresponds to tour 1 in traversal_state
    :return: list[int]
    """

    lock_indices = [len(traversal) - 1 for traversal in traversal_state]
    return lock_indices


def simulated_annealing_with_dynamic_constraints(tours: list[list[any]], nodes: list[Node], distance_matrix: np.array, objective: callable, initial_temperature: int,
                                                 iterations: int, traversal_states: list[list[any]]):
    if not is_feasible(tours, nodes):
        raise InfeasibilityError

    if not is_traversal_ordered_subset_of_tours(tours, traversal_states):
        raise IncorrectTraversalError

    best_objective_function_value = objective(tours, distance_matrix)
    current_tours, current_tours_value = tours, best_objective_function_value

    current_traversal_states = traversal_states
    current_lock_indices = determine_lock_indices(tours, traversal_states)

    i = 0

    while i < iterations:
        # Randomly select an index of the tours. If the tour is completely locked, select a new index until this is not the case
        # TODO: should random extraction be dependent on number of tours or on number of nodes of a tour (normalise probability by number of non-depot nodes)? insertion?
        while True:
            randomly_selected_extraction_tour_index = random.randrange(len(current_tours))
            if current_lock_indices[randomly_selected_extraction_tour_index] < len(current_tours[randomly_selected_extraction_tour_index])-2:
                break

        # Randomly select a node (NOT NODE INDEX!) from lock index onwards from the randomly selected tour
        random_node = random.choice([node for node in current_tours[randomly_selected_extraction_tour_index][current_lock_indices[randomly_selected_extraction_tour_index] + 1:] if node != 0])
        candidate_tours = copy.deepcopy(current_tours)
        candidate_tours[randomly_selected_extraction_tour_index].remove(random_node)

        candidate_traversal_states = copy.deepcopy(current_traversal_states)
        candidate_lock_indices = copy.deepcopy(current_lock_indices)

        # Get rid of empty tours (ASSUMPTION: only the two depot nodes left in the removed tour)
        tmp = len(candidate_tours)
        candidate_tours = [candidate_tour for candidate_tour in candidate_tours if len(candidate_tour) > 2]
        candidate_traversal_states = [candidate_traversal_state for candidate_traversal_state in candidate_traversal_states if len(candidate_traversal_state) > 1]

        # Delete lock_index entry in the case that a node from a [0,n,0] unlocked tour was selected. Maintains order of lock_index
        if tmp - len(candidate_tours):
            candidate_tours.append([0, 0])
            del candidate_lock_indices[randomly_selected_extraction_tour_index]
            candidate_lock_indices.append(0)
        else:
            # Add an empty tour to the end of candidate_tours
            candidate_tours.append([0, 0])
            candidate_lock_indices.append(0)

        # Randomly select an index of the tours for inserting the randomly selected node. Continue selecting until an uncompleted tour is found.
        while True:
            randomly_selected_insertion_tour_index = random.randrange(len(candidate_tours))
            if candidate_lock_indices[randomly_selected_insertion_tour_index]+1 < len(current_tours[randomly_selected_insertion_tour_index]):
                break

        if len(candidate_tours[randomly_selected_insertion_tour_index]) > 2:
            insertion_index = random.randint(candidate_lock_indices[randomly_selected_insertion_tour_index]+1, len(candidate_tours[randomly_selected_insertion_tour_index]) - 1)
        else:
            insertion_index = 1

        candidate_tours[randomly_selected_insertion_tour_index].insert(insertion_index, random_node)
        # Get rid of empty tours (ASSUMPTION: only the two depot nodes left in the removed tour)
        tmp = len(candidate_tours)
        candidate_tours = [candidate_tour for candidate_tour in candidate_tours if len(candidate_tour) > 2]

        # If empty tours were removed, that means that no new tours were created. current_lock_indices must be reverted
        if tmp - len(candidate_tours):
            del candidate_lock_indices[-1]

        if is_feasible(candidate_tours, nodes):
            candidate_tours_value = objective(candidate_tours, distance_matrix)

            # Should this be a LESS or LESSEQUAL?
            if candidate_tours_value <= current_tours_value:
                # Update new best tour
                tours, best_objective_function_value = candidate_tours, candidate_tours_value
                print("Iteration: %d    Distance: %d    Tours: " % (i, candidate_tours_value), candidate_tours, )

                # Possible acceptance based on Metropolis criterion
                difference = candidate_tours_value - current_tours_value

                t = initial_temperature / float(i + 1)

                metropolis = exp(-difference / t)

                if difference < 0 or rand() < metropolis:
                    current_tours, current_tours_value = candidate_tours, candidate_tours_value

                i += 1
        else:
            continue

    # TODO: Return type of best_objective_function_value is a numpy float
    return best_objective_function_value, tours


#simulated_annealing(SIMPLE_TOUR, NODES, DISTANCE_MATRIX, objective, INITIAL_TEMP, ITERATIONS)
simulated_annealing_with_dynamic_constraints(SIMPLE_TOUR, NODES, DISTANCE_MATRIX, objective, INITIAL_TEMP, ITERATIONS, SIMPLE_TOUR_TRAVERSAL_STATE)






