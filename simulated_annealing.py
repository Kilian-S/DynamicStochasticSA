import numpy as np
from global_parameters import DISTANCE_MATRIX, VEHICLE_CAPACITY
from nodes import Node


def objective(x: np.ndarray):
    objective_value = x * DISTANCE_MATRIX

    return np.sum(objective_value)


def check_visitation(tours: list[list[any]], nodes: list[Node]):
    # Flatten the tours list and convert to a set for O(1) lookup time
    visited_nodes = set(node for tour in tours for node in tour)

    # Check that each node (except for node 0) appears in the visited set
    for node in nodes:
        if node.id != 0 and node.id not in visited_nodes:
            return False  # Node was not visited

    # Check that no node appears more than once
    if len(visited_nodes) != len(nodes) - 1:
        return False  # Some nodes were visited more than once

    return True  # All nodes were visited exactly once


def flow_conservation(tours: list[list[int]]) -> bool:
    # Create a dictionary to count how many times each node is visited
    visit_counts = {node_id: 0 for tour in tours for node_id in tour}

    for tour in tours:
        # Increment visit counts for nodes in the tour
        for node_id in tour:
            visit_counts[node_id] += 1

        # Check that all nodes except for the depot are visited exactly once
        for node_id, count in visit_counts.items():
            if node_id != 0 and count != 1:
                return False

    return True  # Flow conservation condition satisfied


def vehicle_capacity(tours: list[list[any]], nodes: list[Node]):
    for tour in tours:
        tour_demand = sum(nodes[node].demand for node in tour)
        if tour_demand > VEHICLE_CAPACITY:
            return False
    return True


def start_at_depot(tours: list[list[any]]):
    for tour in tours:
        if tour[0] != 0:
            return False
    return True


def end_at_depot(tours: list[list[any]]):
    for tour in tours:
        if tour[-1] != 0:
            return False
    return True


def check_feasibility(tours: list[list[any]], nodes: list[Node]):
    return check_visitation(tours, nodes) and flow_conservation() and vehicle_capacity(tours, nodes) and start_at_depot(tours) and end_at_depot(tours)


def simulated_annealing(tours: list[list[any]]):
    pass






