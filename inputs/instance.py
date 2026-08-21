from pathlib import Path
from inputs.distances import read_in_distance_matrix

# The location of the bundled problem instance, resolved from this file so that every script finds it regardless of the working directory it was started from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DISTANCES_FILE = str(PROJECT_ROOT / 'inputs' / 'distances.xlsx')

# The sheet holding the settlement names, coordinates and expected demand
NODES_SHEET = 'Sheet1'

# The sheet and cell range holding the symmetric road distance matrix of the Nurdağı district
DISTANCE_MATRIX_SHEET = 'Distance matrix (districts)'
DISTANCE_MATRIX_TOP_LEFT = 'B2'
DISTANCE_MATRIX_BOTTOM_RIGHT = 'AX50'

# The number of demand nodes in the instance, excluding the depot
NUMBER_OF_NODES = 48

# The vehicle capacity assumed throughout the experiments
VEHICLE_CAPACITY = 2000


def read_instance_distance_matrix():
    """
    Read the road distance matrix of the bundled Nurdağı instance.

    Returns:
        numpy.ndarray: The symmetric distance matrix, in metres.

    """
    return read_in_distance_matrix(DISTANCES_FILE, DISTANCE_MATRIX_SHEET, DISTANCE_MATRIX_TOP_LEFT, DISTANCE_MATRIX_BOTTOM_RIGHT)
