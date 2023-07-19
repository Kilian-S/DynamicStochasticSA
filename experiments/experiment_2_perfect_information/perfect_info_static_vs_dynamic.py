import ast
import pandas as pd
from dynamic_behaviour import initialise_dynamic_data_structures
from inputs.distances import read_in_distance_matrix
from inputs.node import create_nodes_static
from simulated_annealing import is_feasible, objective

vehicle_capacity = 2000

nodes = create_nodes_static('../../inputs/distances.xlsx', 'Sheet1')
distance_matrix = read_in_distance_matrix('../../inputs/distances.xlsx', 'Distance matrix (districts)', 'B2', 'AX50')

dynamic_distance_matrix, dynamic_node_list, node_families, nodes = initialise_dynamic_data_structures(distance_matrix, nodes, vehicle_capacity)

df = pd.read_excel('results_static.xlsx')
df['tours'] = df['tours'].apply(ast.literal_eval)
tours = df['tours'].iloc[0]


def check_static_feasible():
    x = is_feasible(tours, nodes, vehicle_capacity)
    z = objective(tours, dynamic_distance_matrix)

    return x, z





















