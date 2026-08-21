class InfeasibilityError(Exception):
    """
    Indicates an infeasible tour
    """
    pass


class IncorrectTraversalError(Exception):
    """
    Indicates that tour traversals deviate from tours
    """
    pass


class IncorrectReconciliationError(Exception):
    """
    Indicates that tours of the original and new solution have been reconciled incorrectly
    """
    pass


class NodeNotFoundError(Exception):
    """
    Indicates a node is not present in a list
    """
    pass


class SearchStalledError(Exception):
    """
    Indicates that simulated annealing could not generate a feasible candidate solution and would otherwise search indefinitely
    """
    pass
