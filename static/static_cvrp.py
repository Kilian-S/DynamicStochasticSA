import time
import matplotlib.pyplot as plt
from docplex.mp.model import Model
from inputs.distances import normalise_geo_coordinates
from inputs.instance import DISTANCES_FILE, NODES_SHEET, NUMBER_OF_NODES, VEHICLE_CAPACITY, read_instance_distance_matrix
from inputs.node import create_nodes_static

# The coordinates of the south-western corner of the Nurdağı district, used as the origin when plotting
PLOT_ORIGIN = (37.05, 36.65)

# The time the solver is allowed to spend on the model. The instance is not solved to proven optimality within this budget; the incumbent found is used as the benchmark
SOLVER_TIME_LIMIT = 120


def create_feasibility_array(dictionary: dict, vehicle_capacity: int, num_nodes: int) -> list[int]:
    """
    Create a feasibility array based on a dictionary of node demands. The feasibility array makes any typical routing problem with a single depot solvable. It essentially removes
    the single visitation constraint of the CVRP. The values

    Args:
        dictionary (dict): Dictionary mapping node indices to their respective demands.
        vehicle_capacity (int): Capacity of the vehicle.
        num_nodes (int): Total number of nodes.

    Returns:
        list[int]: Feasibility array indicating the number of full vehicle capacities required for each node.

    """
    # Initialise feasibility_array with zeroes
    feasibility_array = [0] * num_nodes

    # Iterate over the dictionary
    for node, demand in dictionary.items():
        # Store the number of full vehicle_capacities required for each node in feasibility_array
        feasibility_array[node - 1] = demand // vehicle_capacity
        # Update the demand in the dictionary to the remainder after division by vehicle_capacity
        dictionary[node] = demand % vehicle_capacity

    return feasibility_array


def tuples_to_tours(active_arcs):
    """
    Convert the arcs selected by the solver into the tours they describe.

    Args:
        active_arcs: The (i, j) arcs that the solver set to one.

    Returns:
        list[list[int]]: The tours, each starting and ending at the depot.

    """
    # Create a dictionary storing successors for each node
    successors = {}
    for (i, j) in active_arcs:
        if i in successors:
            successors[i].append(j)
        else:
            successors[i] = [j]

    # Initialise the tours
    tours = []

    # Start a new tour from each successor of 0 (depot)
    for start_node in successors[0]:
        current_node = start_node
        tour = [0, current_node]  # Start the tour from the depot

        # Continue until we reach back to the depot
        while True:
            next_node = successors[current_node][0]
            if next_node == 0:
                tour.append(0)
                break
            else:
                tour.append(next_node)
                successors[current_node].remove(next_node)
                current_node = next_node

        # Add the completed tour to the list of tours
        tours.append(tour)

    return tours


def get_total_objective_function_value(solver_objective_function_value: float, feasibility_array: list[int], distance_dict: dict) -> float:
    """
    Calculate the total objective function value. Adds back the tours that were removed when making the routing problem feasible

    Args:
        solver_objective_function_value (float): The current value of the objective function from the solver.
        feasibility_array (list[int]): An array indicating the number of additional tours required for each node.
        distance_dict (dict): A dictionary containing distances between nodes.

    Returns:
        float: The updated total objective function value.

    """
    # Iterate over the feasibility array
    for node, num_tours in enumerate(feasibility_array):
        # For each tour, add the round-trip distance from the depot to the node to the objective function value
        for _ in range(num_tours):
            # Adjust node index because feasibility_array is 0-based and node ids in distance_dict are 1-based
            round_trip_distance = distance_dict[(0, node + 1)] + distance_dict[(node + 1, 0)]
            solver_objective_function_value += round_trip_distance

    return solver_objective_function_value


def plot_solution(loc_x: list[float], loc_y: list[float], q: dict, nodes: list[int], active_arcs=None):
    """
    Plot the nodes of the problem instance and, if given, the arcs of a solution.

    Args:
        loc_x (list[float]): The normalised longitude of each node.
        loc_y (list[float]): The normalised latitude of each node.
        q (dict): The demand of each node.
        nodes (list[int]): The demand nodes, excluding the depot.
        active_arcs: The arcs of the solution to draw. Defaults to None, which plots the instance on its own.

    """
    plt.scatter(loc_x[1:], loc_y[1:], c='b')
    for i in nodes:
        plt.annotate('$q_%d=%d$' % (i, q[i]), (loc_x[i] + 2, loc_y[i]))

    if active_arcs is not None:
        for i, j in active_arcs:
            plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], c='g', alpha=0.3)

    plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
    plt.axis('equal')


def exact_algorithm(show_plots: bool = False, log_output: bool = True) -> tuple:
    """
        Solve the Capacitated Vehicle Routing Problem (CVRP) using an exact algorithm. The solving process uses CPLEX solving engine. The code below is an adaptation of
        implementation presented by Hernan Caceres (see 'README - Examples' for more details)

        Args:
            show_plots (bool): Whether to display the instance and solution plots. Defaults to False so that the method can run unattended.
            log_output (bool): Whether to let the solver write its log to stdout. Defaults to True.

        Returns:
            tuple: A tuple containing the total objective function value, the tours and the execution time in seconds.

    """
    start_time = time.time()

    n = NUMBER_OF_NODES
    Q = VEHICLE_CAPACITY
    N = [i for i in range(1, n + 1)]
    V = [0] + N
    nodes = create_nodes_static(DISTANCES_FILE, NODES_SHEET)
    q = {i: nodes[i].expected_demand for i in N}
    feasibility_array = create_feasibility_array(q, Q, n)

    normalised_locations = normalise_geo_coordinates(DISTANCES_FILE, PLOT_ORIGIN)
    loc_x = [location.longitude for location in normalised_locations]
    loc_y = [location.latitude for location in normalised_locations]

    if show_plots:
        plot_solution(loc_x, loc_y, q, N)
        plt.show()

    A = [(i, j) for i in V for j in V]
    distance_matrix = read_instance_distance_matrix()
    assert len(A) == distance_matrix.size, "Number of arcs and entries in distance matrix must be identical."
    c = {(i, j): distance_matrix[i][j] for i, j in A}

    mdl = Model('CVRP')

    x = mdl.binary_var_dict(A, name='x')
    u = mdl.continuous_var_dict(N, ub=Q, name='u')

    mdl.minimize(mdl.sum(c[i, j] * x[i, j] for i, j in A))
    mdl.add_constraints(mdl.sum(x[i, j] for j in V if j != i) == 1 for i in N)
    mdl.add_constraints(mdl.sum(x[i, j] for i in V if i != j) == 1 for j in N)
    mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j], u[i] + q[j] == u[j]) for i, j in A if i != 0 and j != 0)
    mdl.add_constraints(u[i] >= q[i] for i in N)
    mdl.parameters.threads = 1
    mdl.parameters.timelimit = SOLVER_TIME_LIMIT
    solution = mdl.solve(log_output=log_output)

    if solution is None:
        raise RuntimeError(f"CPLEX found no feasible solution within {SOLVER_TIME_LIMIT} seconds")

    active_arcs = [a for a in A if x[a].solution_value > 0.9]

    if show_plots:
        plot_solution(loc_x, loc_y, q, N, active_arcs)
        plt.show()

    tours = tuples_to_tours(active_arcs)

    total_objective_function_value = get_total_objective_function_value(solution.objective_value, feasibility_array, c)

    end_time = time.time()
    execution_time = end_time - start_time

    return total_objective_function_value, tours, execution_time


if __name__ == '__main__':
    objective_function_value, solution_tours, elapsed = exact_algorithm()
    print(f'Objective function value: {objective_function_value}')
    print(f'Tours: {solution_tours}')
    print(f'Execution time: {elapsed}')
