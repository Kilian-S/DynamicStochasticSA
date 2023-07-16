import ast
import pandas as pd
from dynamic_behaviour import initialise_dynamic_data_structures
from inputs.distances import read_in_distance_matrix
from inputs.node import create_nodes, Node
from simulated_annealing import is_feasible, objective


vehicle_capacity = 2000

nodes = create_nodes('../../inputs/distances.xlsx', 'Sheet1')
distance_matrix = read_in_distance_matrix('../../inputs/distances.xlsx', 'Distance matrix (districts)', 'B2', 'AX50')

dynamic_distance_matrix, dynamic_node_list, node_families, nodes = initialise_dynamic_data_structures(distance_matrix, nodes, vehicle_capacity)

df = pd.read_excel('results_static.xlsx')
df['tours'] = df['tours'].apply(ast.literal_eval)
tours = df['tours'].iloc[0]


def check_static_feasible():
    x = is_feasible(tours, nodes, vehicle_capacity)
    z = objective(tours, dynamic_distance_matrix)

    return x, z


def get_total_oversupply(tours: list[list[str]], nodes: list[Node], vehicle_capacity: int) -> int:
    # Create a dictionary for faster node access
    nodes_dict = {str(node.id): node for node in nodes}

    total_oversupply = 0

    for tour in tours:
        # Iterate over each node in the tour
        for node_id in tour:
            # Ignore the depot
            if node_id != '0':
                # Get the node
                node = nodes_dict[node_id]

                # Calculate oversupply if there is any
                oversupply = max(0, vehicle_capacity - node.expected_demand)
                total_oversupply += oversupply

    return total_oversupply


def get_total_undersupply(tours: list[list[str]], nodes: list[Node], vehicle_capacity: int) -> tuple:
    # Create a dictionary for faster node access
    nodes_dict = {str(node.id): node for node in nodes}

    total_undersupply = 0
    undersupplied_tours_count = 0

    for tour in tours:
        # Calculate total demand for the tour
        tour_demand = sum(nodes_dict[node_id].expected_demand for node_id in tour if node_id != '0')

        # If the total demand exceeds the vehicle capacity, it's an undersupply
        if tour_demand > vehicle_capacity:
            undersupply = tour_demand - vehicle_capacity
            total_undersupply += undersupply
            undersupplied_tours_count += 1

    return total_undersupply, undersupplied_tours_count




print(get_total_undersupply(tours, nodes, vehicle_capacity))
























