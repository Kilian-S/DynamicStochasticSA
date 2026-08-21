from dynamic_behaviour import dynamic_sa
from examples.global_parameters import INITIAL_TEMP, ITERATIONS, NODES, SYM_DISTANCE_MATRIX, UTILISATION_TARGET, VEHICLE_CAPACITY
from inputs.instance import DISTANCES_FILE, NODES_SHEET, VEHICLE_CAPACITY as INSTANCE_VEHICLE_CAPACITY, read_instance_distance_matrix
from inputs.node import create_nodes_static
from simulated_annealing import objective


def run_toy_example():
    """
    Solve the five-node example instance. Expected and actual demand diverge sharply in both directions, so the run exercises tour splitting as well as node deletion.

    Returns:
        tuple: The total distance of the solution and its tours.

    """
    current_tours_value, current_tours, execution_time, nodes = dynamic_sa(NODES, SYM_DISTANCE_MATRIX, objective, INITIAL_TEMP, ITERATIONS, VEHICLE_CAPACITY,
                                                                           UTILISATION_TARGET)

    return current_tours_value, current_tours


def run_case_study(initial_temperature: int = 100, iterations: int = 1000, utilisation_target: float = 0.9):
    """
    Solve the bundled Nurdağı instance under the assumption that expected demand turns out to be correct. Pass nodes built by one of the Cauchy loaders instead to introduce
    demand uncertainty.

    Args:
        initial_temperature (int): The starting temperature of the SA algorithm.
        iterations (int): The number of SA iterations per reoptimisation.
        utilisation_target (float): The vehicle utilisation a tour must reach before it leaves the depot.

    Returns:
        tuple: The total distance of the solution and its tours.

    """
    nodes = create_nodes_static(DISTANCES_FILE, NODES_SHEET)
    distance_matrix = read_instance_distance_matrix()

    current_tours_value, current_tours, execution_time, final_nodes = dynamic_sa(nodes, distance_matrix, objective, initial_temperature, iterations, INSTANCE_VEHICLE_CAPACITY,
                                                                                 utilisation_target)

    return current_tours_value, current_tours


if __name__ == '__main__':
    distance, tours = run_toy_example()
    print(f'\nToy example. Total distance: {distance}')
    print(f'Tours: {tours}')
