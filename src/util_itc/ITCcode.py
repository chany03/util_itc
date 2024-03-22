import numpy as np
import math
import warnings
from scipy.optimize import minimize, Bounds


class util_itc:

    def __init__(self, modeltype, choice, amt1, delay1, amt2, delay2):

        self.modeltype, self.choice, self.amt1, self.delay1, self.amt2, self.delay2 = self.__itc_input_checker(modeltype, choice, amt1, delay1, amt2, delay2)
        
        if np.all(self.choice == 1) or np.all(self.choice == 0):
            warnings.warn('All input data is one-sided')

        self.output = []
        self.output.append(self.fit())
        self.output.append(self.modeltype)
        self.output.append(len(choice))

    
    def fit(self):

        estimated_k = []
        average_likelihoods = []

        best_params = []

        bounds = Bounds(0, 2)

        if self.modeltype == 'GH':

            estimated_s = []

            simple_hyperbolic = util_itc('H', self.choice, self.amt1, self.delay1, self.amt2, self.delay2)
            initial_k = simple_hyperbolic.fit()[0]

            for i in np.arange(0, 1, 0.3):
                init = np.array([initial_k, i])
                result = minimize(self.fun, init, method='SLSQP', bounds=bounds)
                estimated_k.append(result.x[0])
                estimated_s.append(result.x[1])
                average_likelihoods.append(-result.fun)

            best_index = average_likelihoods.index(max(average_likelihoods))
            best_k = estimated_k[best_index]
            best_s = estimated_s[best_index]

            best_params.append(best_k)
            best_params.append(best_s)

        elif self.modeltype == 'Q':

            estimated_b = []

            for i in np.arange(0, 1, 0.2):
                for j in np.arange(0, 1, 0.3):
                    init = np.array([i, j])
                    result = minimize(self.fun, init, method='SLSQP', bounds=bounds)
                    estimated_k.append(result.x[0])
                    estimated_b.append(result.x[1])
                    average_likelihoods.append(-result.fun)

            best_index = average_likelihoods.index(max(average_likelihoods))
            best_k = estimated_k[best_index]
            best_b = estimated_b[best_index]

            best_params.append(best_k)
            best_params.append(best_b)

        else: 

            for i in np.arange(0, 1, 0.2):
                init = np.array(i)
                result = minimize(self.fun, init, method='SLSQP', bounds=bounds)
                estimated_k.append(result.x[0])
                average_likelihoods.append(-result.fun)

            best_index = average_likelihoods.index(max(average_likelihoods))
            best_k = estimated_k[best_index]

            best_params.append(best_k)

        return best_params


    def fun(self, params):

        k = params[0]

        if self.modeltype == 'E':
            util1 = self.__exponential(self.amt1, k, self.delay1)
            util2 = self.__exponential(self.amt2, k, self.delay2)

        if self.modeltype == 'H':
            util1 = self.__hyperbolic(self.amt1, k, self.delay1)
            util2 = self.__hyperbolic(self.amt2, k, self.delay2)
        
        if self.modeltype == 'GH':
            s = params[1]
            util1 = self.__generalized_hyperbolic(self.amt1, k, self.delay1, s)
            util2 = self.__generalized_hyperbolic(self.amt2, k, self.delay2, s)

        if self.modeltype == 'Q':
            b = params[1]
            util1 = self.__quasi_hyperbolic(self.amt1, k, self.delay1, b)
            util2 = self.__quasi_hyperbolic(self.amt2, k, self.delay2, b)

        dv = util2 - util1
        dv_choice = -np.where(self.choice == 0, dv, -dv)
        logp = [-np.log(1 + np.exp(dv_choice[i])) if dv_choice[i] < 709 else -dv_choice[i] for i in range(len(dv_choice))]
        return -np.average(logp)


    @staticmethod
    def __exponential(a, k, d):
        return np.multiply(a, np.exp(np.multiply(-k, d)))
    

    @staticmethod
    def __hyperbolic(a, k, d):
        return np.multiply(a, np.divide(1, np.add(1, np.multiply(k, d))))
    

    @staticmethod
    def __generalized_hyperbolic(a, k, d, s):
        return np.multiply(a, np.divide(1, np.power(np.add(1, np.multiply(k, d)), s)))
    

    @staticmethod
    def __quasi_hyperbolic(a, k, d, b):
        return np.multiply(a, np.multiply(b, np.exp(np.multiply(-k, d))))


    @staticmethod
    def __itc_input_checker(modeltype, choice, amt1, delay1, amt2, delay2):

        modeltypes = ['E', 'H', 'GH', 'Q']

        assert (type(modeltype) == str and modeltype.upper() in modeltypes), f'{modeltype} should be a string from the list "E" (exponential), "H" (hyperbolic), "GH" (generalized hyperbolic), and "Q" (quasi hyperbolic)'
        modeltype = modeltype.upper()

        if not isinstance(choice, np.ndarray):
            try:
                choice = np.array(choice)
            except Exception as e:
                raise RuntimeError(f'{choice} should be an array-like; error converting to numpy array: {e}')
            
        try:
            choice = choice.astype(float)
        except Exception as e:
            raise RuntimeError(f'{choice} should only contain numerical values, error casting to float: {e}')

        assert (choice.ndim == 1), f'{choice} should be a vector'
        assert (choice.size > 2), f'{choice} should have at least 3 elements'
        assert (np.all((choice == 0) | (choice == 1))), f'all elements in {choice} should be 1 or 0'

        if not isinstance(amt1, np.ndarray):
            try:
                amt1 = np.array(amt1)
            except Exception as e:
                raise RuntimeError(f'{amt1} should be an array-like; error converting to numpy array: {e}')
            
        try:
            amt1 = amt1.astype(float)
        except Exception as e:
            raise RuntimeError(f'{amt1} should only contain numerical values, error casting to float: {e}')

        assert (type(amt1) == np.ndarray and amt1.ndim == 1), f'{amt1} should be a vector'
        assert (amt1.size > 2), f'{amt1} should have at least 3 elements'
        assert (np.all(amt1 > 0)), f'{amt1} should be positive numbers only'

        if not isinstance(delay1, np.ndarray):
            try:
                delay1 = np.array(delay1)
            except Exception as e:
                raise RuntimeError(f'{delay1} should be an array-like; error converting to numpy array: {e}')
            
        try:
            delay1 = delay1.astype(float)
        except Exception as e:
            raise RuntimeError(f'{delay1} should only contain numerical values, error casting to float: {e}')

        assert (type(delay1) == np.ndarray and delay1.ndim == 1), f'{delay1} should be a vector'
        assert (delay1.size > 2), f'{delay1} should have at least 3 elements'
        assert (np.all(delay1 > 0)), f'{delay1} should be positive numbers only'

        if not isinstance(amt2, np.ndarray):
            try:
                amt2 = np.array(amt2)
            except Exception as e:
                raise RuntimeError(f'{amt2} should be an array-like; error converting to numpy array: {e}')
            
        try:
            amt2 = amt2.astype(float)
        except Exception as e:
            raise RuntimeError(f'{amt2} should only contain numerical values, error casting to float: {e}')

        assert (type(amt2) == np.ndarray and amt2.ndim == 1), f'{amt2} should be a vector'
        assert (amt2.size > 2), f'{amt2} should have at least 3 elements'
        assert (np.all(amt2 > 0)), f'{amt2} should be positive numbers only'

        if not isinstance(delay2, np.ndarray):
            try:
                delay2 = np.array(delay2)
            except Exception as e:
                raise RuntimeError(f'{delay2} should be an array-like; error converting to numpy array: {e}')
            
        try:
            delay2 = delay2.astype(float)
        except Exception as e:
            raise RuntimeError(f'{delay2} should only contain numerical values, error casting to float: {e}')

        assert (type(delay2) == np.ndarray and delay2.ndim == 1), f'{delay2} should be a vector'
        assert (delay2.size > 2), f'{delay2} should have at least 3 elements'
        assert (np.all(delay2 > 0)), f'{delay2} should be positive numbers only'

        assert (choice.size == amt1.size == delay1.size == amt2.size == delay2.size), 'all vectors should have equal size'

        return modeltype, choice, amt1, delay1, amt2, delay2
