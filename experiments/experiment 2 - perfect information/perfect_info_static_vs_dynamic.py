from dynamic_behaviour import dynamic_sa
from experiments.excel import write_to_excel
from inputs.distances import read_in_distance_matrix
from inputs.node import create_nodes
from simulated_annealing import objective
from static.static_cvrp import exact_algorithm


# Static solution
def experiment_static():
    static = exact_algorithm()
    write_to_excel('perfect_info_static_vs_dynamic.xlsx', 'static', 'a1', static)


# Dynamic solutions
def experiment_dynamic():
    initial_temp = 10000
    iterations = 10000
    utilisation_target = 0.9
    vehicle_capacity = 2000

    nodes = create_nodes('../../inputs/distances.xlsx', 'Sheet1')
    distance_matrix = read_in_distance_matrix('../../inputs/distances.xlsx', 'Distance matrix (districts)', 'B2', 'AX50')

    for i in range(0, 5):
        result = dynamic_sa(nodes, distance_matrix, objective, initial_temp, iterations, vehicle_capacity, utilisation_target)
        write_to_excel('perfect_info_static_vs_dynamic.xlsx', 'dynamic (sigmoid)', f'a{i + 2}', result)

    initial_temp = 10000
    iterations = 1000
    utilisation_target = 0.9
    vehicle_capacity = 2000

    nodes = create_nodes('../../inputs/distances.xlsx', 'Sheet1')
    distance_matrix = read_in_distance_matrix('../../inputs/distances.xlsx', 'Distance matrix (districts)', 'B2', 'AX50')

    for i in range(0, 30):
        result = dynamic_sa(nodes, distance_matrix, objective, initial_temp, iterations, vehicle_capacity, utilisation_target)
        write_to_excel('perfect_info_static_vs_dynamic.xlsx', 'dynamic (sigmoid)', f'b{i + 2}', result)

    initial_temp = 1000
    iterations = 100
    utilisation_target = 0.9
    vehicle_capacity = 2000

    nodes = create_nodes('../../inputs/distances.xlsx', 'Sheet1')
    distance_matrix = read_in_distance_matrix('../../inputs/distances.xlsx', 'Distance matrix (districts)', 'B2', 'AX50')

    for i in range(0, 30):
        result = dynamic_sa(nodes, distance_matrix, objective, initial_temp, iterations, vehicle_capacity, utilisation_target)
        write_to_excel('perfect_info_static_vs_dynamic.xlsx', 'dynamic (sigmoid)', f'c{i + 2}', result)

experiment_dynamic()