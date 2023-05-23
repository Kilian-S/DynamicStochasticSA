import numpy as np
from nodes import Node

INITIAL_TEMP = 10
ITERATIONS = 100

# VRP parameters
VEHICLE_CAPACITY = 6000

DISTANCE_MATRIX = np.array([
    [0, 10, 15, 20, 12],
    [10, 0, 35, 25, 44],
    [15, 35, 0, 30, 10],
    [20, 25, 30, 0, 4],
    [20, 14, 30, 30, 0]
])

NODES = [Node(0, 0, 0), Node(1, 2300, 10), Node(2, 3300, 3300), Node(3, 5000, 1000), Node(4, 200, 3000)]

NN_TOUR = [[0, 1, 2, 4, 0], [0, 3, 0]]

SIMPLE_TOUR = [[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0]]
SIMPLE_TOUR_TRAVERSAL_STATE = [[0], [0], [0], [0]]
