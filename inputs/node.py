from openpyxl import load_workbook


class Node:
    """These are the nodes that are seen in the solving process of the SDCVRP."""
    def __init__(self, id, expected_demand):
        self.id = id  # Node ID
        self.expected_demand = expected_demand  # Demand of the node



class InputNode(Node):
    """These nodes are the nodes that are entered as initial parameters of the SDCVRP. They correspond to actual physical nodes."""
    def __init__(self, id, expected_demand, actual_demand=None):
        super().__init__(id, expected_demand)
        self.actual_demand = actual_demand if actual_demand is not None else 0

    def __repr__(self):
        return f"Node {self.id}, Demand: {self.expected_demand}, Actual Demand: {self.actual_demand}"


def create_nodes(filename: str, sheet_name: str):
    workbook = load_workbook(filename=filename)

    # Use the provided sheet name instead of the active sheet
    sheet = workbook[sheet_name]

    nodes = []

    for row in range(2, sheet.max_row + 1):
        id = row - 2
        demand = sheet.cell(row=row, column=3).value

        # Create a Location object and append it to the list
        # TODO: Add stochasticity
        node = InputNode(str(id), demand, demand)
        nodes.append(node)

    return nodes

