import numpy as np
from inputs.node import Node, InputNode

INITIAL_TEMP = 10
ITERATIONS = 5

# VRP parameters
VEHICLE_CAPACITY = 600

DISTANCE_MATRIX = np.array([
    [0, 10, 15, 20, 12],
    [10, 0, 35, 25, 44],
    [15, 35, 0, 30, 10],
    [20, 25, 30, 0, 4],
    [20, 14, 30, 30, 0]
])

SYM_DISTANCE_MATRIX = np.array([
    [0, 10, 15, 20, 12],
    [10, 0, 35, 25, 44],
    [15, 35, 0, 30, 10],
    [20, 25, 30, 0, 4],
    [12, 44, 10, 4, 0]
])

NODES = [InputNode('0', 0, 0), InputNode('1', 1000, 10), InputNode('2', 400, 3300), InputNode('3', 700, 1000), InputNode('4', 200, 3000)]

NN_TOUR = [[0, 1, 2, 4, 0], [0, 3, 0]]

SIMPLE_TOUR = [[0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0]]
SIMPLE_TOUR_TRAVERSAL_STATE = [[0], [0], [0], [0]]
