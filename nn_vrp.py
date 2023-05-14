from global_parameters import DISTANCE_MATRIX, VEHICLE_CAPACITY, NODES
from nodes import Node


def calculate_total_distance(tours: list[list[int]], distance_matrix: list[list[int]]) -> int:
    total_distance = 0
    for tour in tours:
        # Add up the distance between consecutive nodes in the tour
        total_distance += sum(distance_matrix[tour[i-1]][tour[i]] for i in range(1, len(tour)))
    return total_distance


def nearest_neighbour_vrp(distance_matrix: list[list[int]], nodes: list[Node]) -> list[list[int]]:
    # Initialize
    unvisited_nodes = {node.id for node in nodes if node.id != 0}  # Exclude the depot
    tours = []

    while unvisited_nodes:
        tour = [0]  # Start at the depot
        load = 0

        while True:
            current_node = tour[-1]
            next_node = None
            min_distance = float('inf')

            for node_id in unvisited_nodes:
                node = next(node for node in nodes if node.id == node_id)  # Get the node by id
                if node.demand + load <= VEHICLE_CAPACITY and distance_matrix[current_node][node.id] < min_distance:
                    next_node = node_id
                    min_distance = distance_matrix[current_node][node.id]

            if next_node is None:  # No feasible node to visit next, end this tour
                tour.append(0)  # Return to depot
                tours.append(tour)
                break
            else:  # Visit the next node
                tour.append(next_node)
                load += next(node for node in nodes if node.id == next_node).demand
                unvisited_nodes.remove(next_node)

    return tours


print(nearest_neighbour_vrp(DISTANCE_MATRIX, NODES))
print(calculate_total_distance(nearest_neighbour_vrp(DISTANCE_MATRIX, NODES), DISTANCE_MATRIX))



