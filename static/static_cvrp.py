import matplotlib.pyplot as plt
from docplex.mp.model import Model
from inputs.distances import read_in_distance_matrix, normalise_geo_coordinates
from inputs.node import create_nodes_static
import pandas as pd


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
    # Initialize feasibility_array with zeroes
    feasibility_array = [0] * num_nodes

    # Iterate over the dictionary
    for node, demand in dictionary.items():
        # Store the number of full vehicle_capacities required for each node in feasibility_array
        feasibility_array[node - 1] = demand // vehicle_capacity
        # Update the demand in the dictionary to the remainder after division by vehicle_capacity
        dictionary[node] = demand % vehicle_capacity

    return feasibility_array


def tuples_to_tours(active_arcs):
    # Create a dictionary storing successors for each node
    successors = {}
    for (i, j) in active_arcs:
        if i in successors:
            successors[i].append(j)
        else:
            successors[i] = [j]

    # Initialize the tours
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


def exact_algorithm():
    """
        Solve the Capacitated Vehicle Routing Problem (CVRP) using an exact algorithm. The solving process uses CPLEX solving engine. The code below is an adaptation of
        implementation presented by Hernan Caceres (see 'README - Examples' for more details)

        Returns:
            float: The total objective function value.

    """
    n = 48
    Q = 2000
    N = [i for i in range(1, n + 1)]
    V = [0] + N
    nodes = create_nodes_static('../inputs/distances.xlsx', 'Sheet1')
    q = {i: nodes[i].expected_demand for i in N}
    feasibility_array = create_feasibility_array(q, Q, n)

    normalised_locations = normalise_geo_coordinates('../inputs/distances.xlsx', (37.05, 36.65))
    loc_x = [location.longitude for location in normalised_locations]
    loc_y = [location.latitude for location in normalised_locations]

    plt.scatter(loc_x[1:], loc_y[1:], c='b')
    for i in N:
        plt.annotate('$q_%d=%d$' % (i, q[i]), (loc_x[i] + 2, loc_y[i]))
    plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
    plt.axis('equal')

    plt.show()

    A = [(i, j) for i in V for j in V]
    distance_matrix = read_in_distance_matrix("../inputs/distances.xlsx", "Distance matrix (districts)", "B2", "AX50")
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
    mdl.parameters.timelimit = 1
    solution = mdl.solve(log_output=True)

    print(solution)

    solution.solve_status
    active_arcs = [a for a in A if x[a].solution_value > 0.9]

    plt.scatter(loc_x[1:], loc_y[1:], c='b')
    for i in N:
        plt.annotate('$q_%d=%d$' % (i, q[i]), (loc_x[i] + 2, loc_y[i]))
    for i, j in active_arcs:
        plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], c='g', alpha=0.3)
    plt.plot(loc_x[0], loc_y[0], c='r', marker='s')
    plt.axis('equal')

    plt.show()

    tours = tuples_to_tours(active_arcs)
    print(tours)

    total_objective_function_value = get_total_objective_function_value(solution.objective_value, feasibility_array, c)
    print(total_objective_function_value)

    df = pd.DataFrame({
        'total_objective_function_value': [total_objective_function_value],
        'tour': str(tours)
    })
    #df.to_excel('results_static.xlsx', index=False)

    return total_objective_function_value, tours

exact_algorithm()








