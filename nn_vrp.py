import numpy as np

from global_parameters import DISTANCE_MATRIX


def compute_tour_distance(tour, distance_matrix):
    return sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)) + distance_matrix[tour[-1]][tour[0]]


def nearest_neighbour_vrp(distance_matrix, max_stops):
    num_nodes = len(distance_matrix)
    visited = [False]*num_nodes
    visited[0] = True
    tours = []
    total_distance = 0

    while not all(visited):
        current_node = 0
        tour = [current_node]

        for _ in range(max_stops):
            nearest_node = np.argmin([distance_matrix[current_node][j] if not visited[j] else np.inf for j in range(num_nodes)])
            if np.isinf(nearest_node):
                break
            tour.append(nearest_node)
            visited[nearest_node] = True
            current_node = nearest_node

        tours.append(tour)
        total_distance += compute_tour_distance(tour, distance_matrix)

    print(f'Total distance for all tours: {total_distance}')
    return tours


print(nearest_neighbour_vrp(DISTANCE_MATRIX, 3))



