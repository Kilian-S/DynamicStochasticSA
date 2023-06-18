import unittest

import numpy as np

from inputs.node import Node
from simulated_annealing import create_boolean_matrix, simulated_annealing


class TestCreateBooleanMatrix(unittest.TestCase):
    def test_single_tour(self):
        tours = [['0', '1.1', '0']]
        matrix = create_boolean_matrix(tours)

        print(matrix)

    def test_long_tours(self):
        tours = [['0', '1.1', '0'], ['0', '1.2', '2.1', '0'], ['0', '2.2', '2.3', '2.4', '0'], ['0', '3.1', '4.4', '0'], ['0', '1.5', '2.5', '0'],
                 ['0', '1.6', '4.4', '4.5', '5.1', '0'],
                 ['0', '5.2', '1.6', '2.6', '1.7', '1.8', '0'],
                 ['0', '5.3', '2.7', '2.8', '2.9', '5.4', '3.3', '0'],
                 ['0', '1.3', '3.2', '0'], ['0', '4.2', '4.3', '1.4', '0']]

        matrix = create_boolean_matrix(tours)

        print(matrix)













