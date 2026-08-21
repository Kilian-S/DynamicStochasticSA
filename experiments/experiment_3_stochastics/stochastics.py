import ast
from pathlib import Path
from openpyxl import Workbook
import pandas as pd
from dynamic_behaviour import dynamic_sa
from inputs.dynamic_nodes_list import DynamicNodeList
from inputs.instance import DISTANCES_FILE, NODES_SHEET, VEHICLE_CAPACITY, read_instance_distance_matrix
from inputs.node import create_nodes_cauchy_dependent_on_expected_demand, InputNode, create_nodes_static, create_nodes_cauchy, Node
from inputs.node_family import NodeFamily
from simulated_annealing import objective

INITIAL_TEMPERATURE = 100
ITERATIONS = 1000
UTILISATION_TARGET = 0.9

# Results are written next to this script rather than into whichever working directory the script happens to be started from
OUTPUT_DIRECTORY = Path(__file__).resolve().parent

# The tours of the exact solution, against which the dynamic solution is compared. This is the copy whose tours have been reindexed by child node, which is the basis the
# omniscient nodes are expressed in; the raw solver output in results_static.xlsx indexes tours by node family and cannot be compared against them directly
EXACT_SOLUTION_FILE = OUTPUT_DIRECTORY.parent / 'experiment_2_perfect_information' / 'results_static_simple.xlsx'


def get_omniscient_nodes(stochastic_nodes: list[InputNode], vehicle_capacity: int):
    """
    Determine the child nodes that the problem instance would consist of if every actual demand were known in advance. These are the nodes against which a solution's service
    level, oversupply and undersupply are measured.

    Args:
        stochastic_nodes (list[InputNode]): The input nodes, carrying both expected and actual demand.
        vehicle_capacity (int): The vehicle capacity.

    Returns:
        list[Node]: The child nodes implied by the actual demand of every node family.

    """
    node_families = [NodeFamily(node, vehicle_capacity) for node in stochastic_nodes]
    dynamic_node_list = DynamicNodeList(node_families, vehicle_capacity)
    for node_family in node_families[1:]:
        node_family.update()
    omniscient_nodes = dynamic_node_list.get_all_nodes()
    return omniscient_nodes


def get_total_oversupply(tours: list[list[str]], nodes: list[Node], vehicle_capacity: int, tours_set: set[str], nodes_set: set[str]) -> int:
    """
    Calculate the total unused vehicle capacity across all tours. A node that a solution never serves is counted as a tour that was never dispatched, and so as a full vehicle
    load of unused capacity less whatever that node would have taken.

    Args:
        tours (list[list[str]]): The tours of the solution.
        nodes (list[Node]): The omniscient nodes.
        vehicle_capacity (int): The vehicle capacity.
        tours_set (set[str]): The IDs of the nodes the solution visits.
        nodes_set (set[str]): The IDs of the omniscient nodes.

    Returns:
        int: The total oversupply.

    """
    # Create a dictionary for faster node access
    nodes_dict = {node.id: node for node in nodes}

    total_oversupply = 0

    for tour in tours:
        tour_demand = 0

        for node_id in tour:
            node = nodes_dict.get(node_id)
            if node is not None:
                tour_demand += node.expected_demand

        tour_oversupply = max(0, vehicle_capacity - tour_demand)
        total_oversupply += tour_oversupply

    # Handle nodes not considered by the exact solution
    if tours_set != nodes_set:
        difference_set = nodes_set - tours_set

        for node_id in difference_set:
            # A child node never carries more than one vehicle load, so this term cannot go negative
            total_oversupply += max(0, vehicle_capacity - nodes_dict.get(node_id).expected_demand)

    return total_oversupply


def get_total_undersupply(tours: list[list[str]], nodes: list[Node], vehicle_capacity: int, tours_set: set[str], nodes_set: set[str]) -> tuple:
    """
    Calculate the total demand that a solution fails to serve, and the number of tours on which it falls short. A node the solution never visits contributes its entire demand.

    Args:
        tours (list[list[str]]): The tours of the solution.
        nodes (list[Node]): The omniscient nodes.
        vehicle_capacity (int): The vehicle capacity.
        tours_set (set[str]): The IDs of the nodes the solution visits.
        nodes_set (set[str]): The IDs of the omniscient nodes.

    Returns:
        tuple: The total undersupply and the number of tours that are undersupplied.

    """
    # Create a dictionary for faster node access
    nodes_dict = {node.id: node for node in nodes}
    total_undersupply = 0
    undersupplied_tours_count = 0

    for tour in tours:
        tour_demand = 0

        for node_id in tour:
            node = nodes_dict.get(node_id)
            if node is not None:
                tour_demand += node.expected_demand

        if tour_demand > vehicle_capacity:
            total_undersupply += tour_demand - vehicle_capacity
            undersupplied_tours_count += 1

    # Handle nodes not considered by the exact solution
    if tours_set != nodes_set:
        difference_set = nodes_set - tours_set

        for node_id in difference_set:
            total_undersupply += nodes_dict.get(node_id).expected_demand
            undersupplied_tours_count += 1

    return total_undersupply, undersupplied_tours_count


def get_service_level(tours: list[list[str]], nodes: list[Node], vehicle_capacity: int, tours_set: set[str], nodes_set: set[str]) -> float:
    """
    Calculate the proportion of total network demand that a solution actually serves. A vehicle serves the nodes of its tour in order until it runs out of goods; demand beyond
    that point, and the demand of any node the solution never visits, goes unserved.

    Args:
        tours (list[list[str]]): The tours of the solution.
        nodes (list[Node]): The omniscient nodes.
        vehicle_capacity (int): The vehicle capacity.
        tours_set (set[str]): The IDs of the nodes the solution visits.
        nodes_set (set[str]): The IDs of the omniscient nodes.

    Returns:
        float: The service level, between 0 and 1.

    """
    total_network_demand = 0
    served_demand = 0
    nodes_dict = {node.id: node for node in nodes}

    for tour in tours:
        tour_demand = 0  # Track total demand for this tour
        remaining_capacity = vehicle_capacity  # Reset the remaining vehicle capacity for each tour

        for node_id in tour:
            node = nodes_dict.get(node_id)
            if node is not None:
                node_demand = node.expected_demand
                tour_demand += node_demand

                # Count the node's demand towards served demand only if there's still capacity left
                if remaining_capacity > 0:
                    demand_served_at_this_node = min(node_demand, remaining_capacity)
                    served_demand += demand_served_at_this_node
                    remaining_capacity -= demand_served_at_this_node

        total_network_demand += tour_demand

    # Add demand of nodes not considered in the static solution
    if tours_set != nodes_set:
        difference_set = nodes_set - tours_set
        for node_id in difference_set:
            node = nodes_dict.get(node_id)
            if node is not None:
                total_network_demand += node.expected_demand

    # Avoid division by zero error
    if total_network_demand == 0:
        return 0.0

    return served_demand / total_network_demand


def read_exact_tours(exact_solution_file=EXACT_SOLUTION_FILE) -> list[list[str]]:
    """
    Read the tours of the exact solution that the dynamic solution is benchmarked against.

    Args:
        exact_solution_file: The Excel file holding the results of the exact algorithm.

    Returns:
        list[list[str]]: The tours of the exact solution.

    """
    df = pd.read_excel(exact_solution_file)
    df['tours'] = df['tours'].apply(ast.literal_eval)
    return df['tours'].iloc[0]


def stochastic_nodes_to_excel(n: int, gamma_values: list[float], output_filename='comparison.xlsx', filename=DISTANCES_FILE, sheetname=NODES_SHEET):
    """
    Generate n sets of Cauchy distributed demands for each gamma value and write them to an Excel file, one sheet per gamma value. The deterministic demands are written as the
    first row of every sheet so that the spread of each sample can be read off against them.

    Args:
        n (int): The number of demand sets generated per gamma value.
        gamma_values (list[float]): The Cauchy scale parameters under test.
        output_filename: The Excel file the demands are written to.
        filename: The workbook holding the problem instance.
        sheetname: The sheet holding the node data.

    """
    static_nodes = create_nodes_static(filename, sheetname)

    workbook = Workbook()

    # Remove default sheet if it exists
    if 'Sheet' in workbook.sheetnames:
        del workbook['Sheet']

    # Writing the static nodes to the Excel file
    static_demand = [node.actual_demand for node in static_nodes]

    # Generating n sets of nodes with Cauchy distributed demand for each gamma value
    for gamma in gamma_values:
        # Create a new sheet for this gamma value
        sheet = workbook.create_sheet(title=f"Gamma_{gamma}")
        sheet.append(static_demand)

        for _ in range(n):
            stochastic_nodes = create_nodes_cauchy(filename, sheetname, gamma)
            stochastic_demand = [node.actual_demand for node in stochastic_nodes]
            sheet.append(stochastic_demand)

    workbook.save(OUTPUT_DIRECTORY / output_filename)


def stochastic_nodes_to_excel_dependent_on_expected_demand(n: int, gamma_factors: list[float], output_filename='comparison_normalised.xlsx', filename=DISTANCES_FILE,
                                                           sheetname=NODES_SHEET):
    """
    As stochastic_nodes_to_excel, but with a Cauchy scale parameter proportional to each node's expected demand rather than shared across all nodes.

    Args:
        n (int): The number of demand sets generated per gamma factor.
        gamma_factors (list[float]): The proportions of expected demand used as Cauchy scale parameters.
        output_filename: The Excel file the demands are written to.
        filename: The workbook holding the problem instance.
        sheetname: The sheet holding the node data.

    """
    static_nodes = create_nodes_static(filename, sheetname)

    workbook = Workbook()

    # Remove default sheet if it exists
    if 'Sheet' in workbook.sheetnames:
        del workbook['Sheet']

    # Writing the static nodes to the Excel file
    static_demand = [node.actual_demand for node in static_nodes]

    # Generating n sets of nodes with Cauchy distributed demand for each gamma factor
    for gamma_factor in gamma_factors:
        # Create a new sheet for this gamma factor value
        sheet = workbook.create_sheet(title=f"Gamma_Factor_{gamma_factor}")
        sheet.append(static_demand)

        for _ in range(n):
            stochastic_nodes = create_nodes_cauchy_dependent_on_expected_demand(filename, sheetname, gamma_factor)
            stochastic_demand = [node.actual_demand for node in stochastic_nodes]
            sheet.append(stochastic_demand)

    workbook.save(OUTPUT_DIRECTORY / output_filename)


def run_stochastic_experiment(trials: int, gammas: list[float], output_filename: str, create_stochastic_nodes: callable, gamma_column: str = 'gamma_factor'):
    """
        Compare the dynamic solution against the exact solution across a range of demand volatilities. For every trial a fresh set of actual demands is drawn, the dynamic
        solution is recomputed from scratch, and both solutions are scored against the demands that were actually revealed.

        Args:
            trials (int): The number of trials run per gamma value.
            gammas (list[float]): The demand volatilities under test.
            output_filename (str): The name of the Excel file the results are written to.
            create_stochastic_nodes (callable): A function mapping a gamma value to a fresh set of input nodes.
            gamma_column (str): The name of the column recording the gamma value.

    """
    distance_matrix = read_instance_distance_matrix()

    # Exact solution
    exact_tours = read_exact_tours()
    exact_tours_set = set(string for tour in exact_tours for string in tour)

    # Create a dictionary to store data for the DataFrame
    data = {gamma_column: [], 'trial': [], 'demands': [], 'objective_value': [], 'tours': [], 'execution_time': [], 'service_level_sa': [], 'total_oversupply_sa': [],
            'total_undersupply_sa': [], 'total_undersupply_sa_count': [], 'service_level_exact': [], 'total_oversupply_exact': [], 'total_undersupply_exact': [],
            'total_undersupply_exact_count': []}

    for gamma in gammas:
        for trial in range(trials):
            stochastic_nodes = create_stochastic_nodes(gamma)
            stochastic_demand = [node.actual_demand for node in stochastic_nodes]
            current_tours_value, current_tours, execution_time, omniscient_nodes = dynamic_sa(stochastic_nodes, distance_matrix, objective, INITIAL_TEMPERATURE, ITERATIONS,
                                                                                              VEHICLE_CAPACITY, UTILISATION_TARGET)

            current_tours_set = set(string for tour in current_tours for string in tour)
            omniscient_nodes_set = set(node.id for node in omniscient_nodes)

            service_level_sa = get_service_level(current_tours, omniscient_nodes, VEHICLE_CAPACITY, current_tours_set, omniscient_nodes_set)
            total_oversupply_sa = get_total_oversupply(current_tours, omniscient_nodes, VEHICLE_CAPACITY, current_tours_set, omniscient_nodes_set)
            total_undersupply_sa, total_undersupply_sa_count = get_total_undersupply(current_tours, omniscient_nodes, VEHICLE_CAPACITY, current_tours_set, omniscient_nodes_set)
            service_level_exact = get_service_level(exact_tours, omniscient_nodes, VEHICLE_CAPACITY, exact_tours_set, omniscient_nodes_set)
            total_oversupply_exact = get_total_oversupply(exact_tours, omniscient_nodes, VEHICLE_CAPACITY, exact_tours_set, omniscient_nodes_set)
            total_undersupply_exact, total_undersupply_exact_count = get_total_undersupply(exact_tours, omniscient_nodes, VEHICLE_CAPACITY, exact_tours_set, omniscient_nodes_set)

            # Append data to the dictionary
            data[gamma_column].append(gamma)
            data['trial'].append(trial + 1)
            data['demands'].append(stochastic_demand)
            data['objective_value'].append(current_tours_value)
            data['tours'].append(str(current_tours))
            data['execution_time'].append(execution_time)
            data['service_level_sa'].append(service_level_sa)
            data['total_oversupply_sa'].append(total_oversupply_sa)
            data['total_undersupply_sa'].append(total_undersupply_sa)
            data['total_undersupply_sa_count'].append(total_undersupply_sa_count)
            data['service_level_exact'].append(service_level_exact)
            data['total_oversupply_exact'].append(total_oversupply_exact)
            data['total_undersupply_exact'].append(total_undersupply_exact)
            data['total_undersupply_exact_count'].append(total_undersupply_exact_count)

            # Write out after every trial so that a long run can be inspected while it is still going
            df = pd.DataFrame(data)
            df.to_excel(OUTPUT_DIRECTORY / output_filename, index=False)


def stochastic_nodes_experiment(trials: int, gamma_values: list[float], output_filename: str):
    """
    Run the comparison with a Cauchy scale parameter that is the same for every node, so that every settlement carries the same absolute uncertainty.

    Args:
        trials (int): The number of trials run per gamma value.
        gamma_values (list[float]): The Cauchy scale parameters under test.
        output_filename (str): The name of the Excel file the results are written to.

    """
    run_stochastic_experiment(trials, gamma_values, output_filename, lambda gamma: create_nodes_cauchy(DISTANCES_FILE, NODES_SHEET, gamma), 'gamma_value')


def stochastic_nodes_dependent_on_expected_demand_experiment(trials: int, gamma_factors: list[float], output_filename: str):
    """
    Run the comparison with a Cauchy scale parameter proportional to each node's expected demand, so that larger settlements carry proportionally more uncertainty. These are the
    reported results.

    Args:
        trials (int): The number of trials run per gamma factor.
        gamma_factors (list[float]): The proportions of expected demand used as Cauchy scale parameters.
        output_filename (str): The name of the Excel file the results are written to.

    """
    run_stochastic_experiment(trials, gamma_factors, output_filename,
                              lambda gamma_factor: create_nodes_cauchy_dependent_on_expected_demand(DISTANCES_FILE, NODES_SHEET, gamma_factor), 'gamma_factor')


if __name__ == '__main__':
    stochastic_nodes_dependent_on_expected_demand_experiment(50, [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15], 'results_stochastics_cauchy_gamma_factors.xlsx')
