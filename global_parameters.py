import numpy as np

# Simulated annealing algorithm
INITIAL_TEMP = 1000


# VRP parameters
VEHICLE_CAPACITY = 6000

DISTANCE_MATRIX = np.array([
    [0, 10, 15, 20, 12],
    [10, 0, 35, 25, 44],
    [15, 35, 0, 30, 10],
    [20, 25, 30, 0,  4],
    [20, 14, 30, 30, 0]
])
