from inputs.node_family import NodeFamily


class DynamicNodeList:
    def __init__(self, node_families: list[NodeFamily], vehicle_capacity: int):
        self.node_families = node_families
        self.vehicle_capacity = vehicle_capacity

    def get_all_nodes(self):
        """Return a list of all child nodes in all node families."""
        return [node for node_family in self.node_families for node in node_family.child_nodes]

    def get_all_child_nodes_from_family(self, node_family_id: str):
        for node_family in self.node_families:
            if node_family.node_family_id == node_family_id:
                return node_family.child_nodes

