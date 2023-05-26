from openpyxl import load_workbook


class Node:
    def __init__(self, id, expected_demand, actual_demand=None):
        self.id = id  # Node ID
        self.expected_demand = expected_demand  # Demand of the node
        self.actual_demand = actual_demand

        if actual_demand is None:
            actual_demand = 0

    def __repr__(self):
        return f"Node {self.id}, Demand: {self.expected_demand}"


def create_nodes(f):
    workbook = load_workbook(filename=f)
    sheet = workbook.active
    nodes = []

    for row in range(2, sheet.max_row + 1):
        id = row - 2
        demand = sheet.cell(row=row, column=3).value

        # Create a Location object and append it to the list
        node = Node(id, demand)
        nodes.append(node)

    return nodes
