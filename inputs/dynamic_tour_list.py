from inputs.node import Node


def create_tours(tours: list[list[str]], nodes: list[Node]) -> list[list[Node]]:
    # Create a dictionary that maps node ids to nodes
    node_dict = {node.id: node for node in nodes}

    # Use list comprehension to create the new tours
    new_tours = [[node_dict[node_id] for node_id in tour] for tour in tours]

    return new_tours


class DynamicTourList:
    def __init__(self, initial_solution_tours: list[list[str]], nodes: list[Node]):
        self.tours = create_tours(initial_solution_tours, nodes)

    def __repr__(self):
        return f"Tours: {self.tours}"
