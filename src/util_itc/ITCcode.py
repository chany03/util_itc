import numpy as np
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class util_itc:

    def __init__(self, modeltype, choice, amt1, delay1, amt2, delay2):
        
        itc_input_checker(modeltype, choice, amt1, delay1, amt2, delay2)

        self.modeltype = modeltype
        self.choice = choice
        self.amt1 = amt1
        self.delay1 = delay1
        self.amt2 = amt2
        self.delay2 = delay2

def itc_input_checker(modeltype, choice, amt1, delay1, amt2, delay2):

        modeltypes = ['E', 'H', 'GH', 'Q']

        assert (type(modeltype) == str and modeltype in modeltypes), f'{modeltype} should be a string from the list "E" (exponential), "H" (hyperbolic), "GH" (generalized hyperbolic), and "Q" (quasi hyperbolic)'

        assert (type(choice) == np.ndarray and choice.ndim == 1), f'{choice} should be a vector'
        assert (choice.size > 2), f'{choice} should have at least 3 elements'
        assert (np.all((choice == 0) | (choice == 1))), f'all elements in {choice} should be 1 or 0'

        assert (type(amt1) == np.ndarray and amt1.ndim == 1), f'{amt1} should be a vector'
        assert (amt1.size > 2), f'{amt1} should have at least 3 elements'
        assert (np.all(amt1 > 0)), f'{amt1} should be positive numbers only'

        assert (type(delay1) == np.ndarray and delay1.ndim == 1), f'{delay1} should be a vector'
        assert (delay1.size > 2), f'{delay1} should have at least 3 elements'
        assert (np.all(delay1 > 0)), f'{delay1} should be positive numbers only'

        assert (type(amt2) == np.ndarray and amt2.ndim == 1), f'{amt2} should be a vector'
        assert (amt2.size > 2), f'{amt2} should have at least 3 elements'
        assert (np.all(amt2 > 0)), f'{amt2} should be positive numbers only'

        assert (type(delay2) == np.ndarray and delay2.ndim == 1), f'{delay2} should be a vector'
        assert (delay2.size > 2), f'{delay2} should have at least 3 elements'
        assert (np.all(delay2 > 0)), f'{delay2} should be positive numbers only'

        assert (choice.size == amt1.size == delay1.size == amt2.size == delay2.size), 'all vectors should have equal size'

        return 'Input check completed successfully.'