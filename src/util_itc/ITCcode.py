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

        assert (type(modeltype) == str and modeltype in modeltypes), modeltype + ' should be a string from the list "E" (exponential), "H" (hyperbolic), "GH" (generalized hyperbolic), and "Q" (quasi hyperbolic)'

        assert (type(choice) == int and choice in [0, 1]), choice + ' should be 1 for option 1 or 0 for option 2'

        a, b = amt1.shape
        assert (type(amt1) == np.ndarray and (a == 1 or b == 1)), amt1 + ' should be a vector'
        assert (amt1.size > 2), amt1 + ' should have at least 3 elements'
        assert (np.all(amt1 > 0)), amt1 + ' should be positive numbers only'

        c, d = delay1.shape
        assert (type(delay1) == np.ndarray and (c == 1 or d == 1)), delay1 + ' should be a vector'
        assert (delay1.size > 2), delay1 + ' should have at least 3 elements'
        assert (np.all(delay1 > 0)), delay1 + ' should be positive numbers only'

        e, f = amt2.shape
        assert (type(amt2) == np.ndarray and (e == 1 or f == 1)), amt2 +  'should be a vector'
        assert (amt2.size > 2), amt2 + ' should have at least 3 elements'
        assert (np.all(amt2 > 0)), amt2 + ' should be positive numbers only'

        g, h = delay2.shape
        assert (type(delay2) == np.ndarray and (g == 1 or h == 1)), delay2 +  'should be a vector'
        assert (delay2.size > 2), delay2 + ' should have at least 3 elements'
        assert (np.all(delay2 > 0)), delay2 + ' should be positive numbers only'

        return 'Input check completed successfully.'