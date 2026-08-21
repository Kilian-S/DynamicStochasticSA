from openpyxl import load_workbook
import numpy as np


class Node:
    """These are the nodes that are seen in the solving process of the SDCVRP."""
    def __init__(self, id, expected_demand):
        self.id = id  # Node ID
        self.expected_demand = expected_demand  # Demand of the node

    def __eq__(self, other):
        # Nodes are identified by their ID alone. Two nodes with the same ID describe the same visit, even if their expected demand has since been revised
        if not isinstance(other, Node):
            return NotImplemented
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"Node {self.id}, Demand: {self.expected_demand}"


class InputNode(Node):
    """These nodes are the nodes that are entered as initial parameters of the SDCVRP. They correspond to actual physical nodes."""
    def __init__(self, id, expected_demand, actual_demand=None):
        super().__init__(id, expected_demand)
        self.actual_demand = actual_demand if actual_demand is not None else 0

    def __repr__(self):
        return f"Node {self.id}, Demand: {self.expected_demand}, Actual Demand: {self.actual_demand}"


def read_expected_demands(filename: str, sheet_name: str) -> list[int]:
    """
    Read the expected demand of every node from the given worksheet. The first data row is assumed to be the depot.

    Args:
        filename (str): The name of the workbook file to load.
        sheet_name (str): The name of the sheet holding the node data.

    Returns:
        list[int]: The expected demand of each node, indexed by node ID.

    """
    workbook = load_workbook(filename=filename, read_only=True)

    # Use the provided sheet name instead of the active sheet
    sheet = workbook[sheet_name]

    expected_demands = []

    for row in range(2, sheet.max_row + 1):
        expected_demand = sheet.cell(row=row, column=3).value

        # An empty cell is treated as no demand rather than propagating a None into the solver
        expected_demands.append(0 if expected_demand is None else int(expected_demand))

    workbook.close()

    return expected_demands


def create_nodes_static(filename: str, sheet_name: str) -> list[InputNode]:
    """
    Create the input nodes of a deterministic problem instance. Actual demand is set equal to expected demand, so no demand is revealed during the solving process.

    Args:
        filename (str): The name of the workbook file to load.
        sheet_name (str): The name of the sheet holding the node data.

    Returns:
        list[InputNode]: The input nodes of the problem instance.

    """
    nodes = []

    for node_id, expected_demand in enumerate(read_expected_demands(filename, sheet_name)):
        actual_demand = expected_demand

        # Create an InputNode object and append it to the list
        node = InputNode(str(node_id), expected_demand, actual_demand)
        nodes.append(node)

    return nodes


def create_nodes_from_cauchy_scales(expected_demands: list[int], gammas: list[float]) -> list[InputNode]:
    """
    Create the input nodes of a stochastic problem instance. The actual demand of each node (except for the depot) is drawn from a Cauchy distribution centred on its expected
    demand.

    Args:
        expected_demands (list[int]): The expected demand of each node, indexed by node ID.
        gammas (list[float]): The Cauchy scale parameter of each node, indexed by node ID.

    Returns:
        list[InputNode]: The input nodes of the problem instance.

    """
    nodes = []

    for node_id, (expected_demand, gamma) in enumerate(zip(expected_demands, gammas)):
        if node_id == 0:
            nodes.append(InputNode('0', 0, 0))
            continue

        # Generate actual_demand from a Cauchy distribution
        actual_demand = np.random.standard_cauchy() * gamma + expected_demand

        # Ensure that actual_demand is non-negative
        actual_demand = max(0, actual_demand)

        # Create an InputNode object and append it to the list
        node = InputNode(str(node_id), expected_demand, int(actual_demand))
        nodes.append(node)

    return nodes


def create_nodes_cauchy(filename: str, sheet_name: str, gamma: float) -> list[InputNode]:
    """
    Create the input nodes of a stochastic problem instance in which every node shares the same Cauchy scale parameter.

    Args:
        filename (str): The name of the workbook file to load.
        sheet_name (str): The name of the sheet holding the node data.
        gamma (float): The Cauchy scale parameter applied to every node.

    Returns:
        list[InputNode]: The input nodes of the problem instance.

    """
    expected_demands = read_expected_demands(filename, sheet_name)
    return create_nodes_from_cauchy_scales(expected_demands, [gamma] * len(expected_demands))


def create_nodes_cauchy_dependent_on_expected_demand(filename: str, sheet_name: str, gamma_factor: float) -> list[InputNode]:
    """
    Create the input nodes of a stochastic problem instance in which the Cauchy scale parameter of each node is proportional to its expected demand. Larger nodes therefore carry
    proportionally more uncertainty.

    Args:
        filename (str): The name of the workbook file to load.
        sheet_name (str): The name of the sheet holding the node data.
        gamma_factor (float): The proportion of expected demand used as the Cauchy scale parameter.

    Returns:
        list[InputNode]: The input nodes of the problem instance.

    """
    expected_demands = read_expected_demands(filename, sheet_name)
    return create_nodes_from_cauchy_scales(expected_demands, [expected_demand * gamma_factor for expected_demand in expected_demands])
