from inputs.node import Node


class Tour:
    def __init__(self, tour: list[Node]):
        self.tour = tour

    def get_tour_demand(self):
        return sum(node.expected_demand for node in self.tour)
