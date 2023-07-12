from inputs.node_family import NodeFamily


class DynamicNodeList:
    """
    The DynamicNodeList (DNL) manages the current node families. Every physical node in the CVPR corresponds to a separate node family. However, when nodes are visited more than
    once, a child node of the physical nodes' node family is created. To manage the dynamic growth and contraction of the number of nodes a DNL is required.
    """
    def __init__(self, node_families: list[NodeFamily], vehicle_capacity: int):
        """
        Initialize a DynamicNodeList object.

        Args:
            node_families (list[NodeFamily]): A list of NodeFamily objects representing the node families.
            vehicle_capacity (int): The capacity of the vehicles used in the CVRP.
        """
        self.node_families = node_families
        self.vehicle_capacity = vehicle_capacity

    def get_all_nodes(self):
        """Return a list of all child nodes in all node families."""
        return [node for node_family in self.node_families for node in node_family.child_nodes]

    def get_all_child_nodes_from_family(self, node_family_id: str):
        """
        Retrieve all child nodes associated with a specific node family.

        Args:
            node_family_id (str): The ID of the node family.

        Returns:
            list[Node]: A list of child nodes belonging to the specified node family.
        """
        for node_family in self.node_families:
            if node_family.node_family_id == node_family_id:
                return node_family.child_nodes

    def __repr__(self):
        return f"DNL managing {len(self.node_families)} node families and assuming a vehicle capacity of {self.vehicle_capacity}"


