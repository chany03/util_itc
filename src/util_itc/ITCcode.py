import numpy as np
import math
import warnings
from scipy.optimize import minimize, Bounds


class util_itc:

    def __init__(self, modeltype, choice, amt1, delay1, amt2, delay2):

        self.modeltype, self.choice, self.amt1, self.delay1, self.amt2, self.delay2 = self.__itc_input_checker(modeltype, choice, amt1, delay1, amt2, delay2)
        
        if np.all(self.choice == 1) or np.all(self.choice == 0):
            warnings.warn('All input data is one-sided')

        self.params = self.fit()

        self.output = []
        self.output.append(self.params)
        self.output.append(self.modeltype)
        self.output.append(len(choice))

        self.fun(self.params, onesided_test=True)

    
    def fit(self):

        estimated_k = []
        estimated_it = []
        average_likelihoods = []

        best_params = []

        if self.modeltype == 'GH':

            lower_k, upper_k = self.__bounds_h(self.amt1, self.delay1, self.amt2, self.delay2)
            lower_k_log = math.log(lower_k)
            upper_k_log = math.log(upper_k)

            bounds = Bounds([lower_k_log, -1, -1], [upper_k_log, 2, 2])

            estimated_s = []

            simple_hyperbolic = util_itc('H', self.choice, self.amt1, self.delay1, self.amt2, self.delay2)
            initial_k = simple_hyperbolic.fit()[0]
            initial_it = simple_hyperbolic.fit()[1]

            for i in np.arange(0, 1, 0.3):
                init = np.array([initial_k, initial_it, i])
                result = minimize(self.fun, init, method='SLSQP', bounds=bounds)
                estimated_k.append(math.exp(result.x[0]))
                estimated_it.append(math.exp(result.x[1]))
                estimated_s.append(math.exp(result.x[2]))
                average_likelihoods.append(-result.fun)

            best_index = average_likelihoods.index(max(average_likelihoods))
            best_k = estimated_k[best_index]
            best_it = estimated_it[best_index]
            best_s = estimated_s[best_index]

            best_params.append(best_k)
            best_params.append(best_it)
            best_params.append(best_s)

        elif self.modeltype == 'Q':

            lower_k, upper_k = self.__bounds_e(self.amt1, self.delay1, self.amt2, self.delay2)
            lower_k_log = math.log(lower_k)
            upper_k_log = math.log(upper_k)

            bounds = Bounds([lower_k_log, -1, 0], [upper_k_log, 2, 1])

            estimated_b = []

            for i in np.arange(0, 1, 0.2):
                for j in np.arange(0, 1, 0.3):
                    init = np.array([i, 1, j])
                    result = minimize(self.fun, init, method='SLSQP', bounds=bounds)
                    estimated_k.append(math.exp(result.x[0]))
                    estimated_it.append(math.exp(result.x[1]))
                    estimated_b.append(result.x[2])
                    average_likelihoods.append(-result.fun)

            best_index = average_likelihoods.index(max(average_likelihoods))
            best_k = estimated_k[best_index]
            best_it = estimated_it[best_index]
            best_b = estimated_b[best_index]

            best_params.append(best_k)
            best_params.append(best_it)
            best_params.append(best_b)

        else: 

            if self.modeltype == 'E':
                lower_k, upper_k = self.__bounds_e(self.amt1, self.delay1, self.amt2, self.delay2)
            else:
                lower_k, upper_k = self.__bounds_h(self.amt1, self.delay1, self.amt2, self.delay2)

            lower_k_log = math.log(lower_k)
            upper_k_log = math.log(upper_k)

            bounds = Bounds([lower_k_log, -1], [upper_k_log, 2])

            for i in np.arange(0, 2, 0.2):
                for j in np.arange(0, 3, 0.3):
                    init = np.array([i, j])
                    result = minimize(self.fun, init, method='SLSQP', bounds=bounds)
                    estimated_k.append(math.exp(result.x[0]))
                    estimated_it.append(math.exp(result.x[1]))
                    average_likelihoods.append(-result.fun)

            best_index = average_likelihoods.index(max(average_likelihoods))
            best_k = estimated_k[best_index]
            best_it = estimated_it[best_index]

            best_params.append(best_k)
            best_params.append(best_it)

        return best_params


    def fun(self, params, onesided_test=False):

        k = math.exp(params[0])
        inverse_temp = math.exp(params[1])

        if self.modeltype == 'E':
            util1 = self.__exponential(self.amt1, k, self.delay1)
            util2 = self.__exponential(self.amt2, k, self.delay2)

        if self.modeltype == 'H':
            util1 = self.__hyperbolic(self.amt1, k, self.delay1)
            util2 = self.__hyperbolic(self.amt2, k, self.delay2)
        
        if self.modeltype == 'GH':
            s = math.exp(params[2])
            util1 = self.__generalized_hyperbolic(self.amt1, k, self.delay1, s)
            util2 = self.__generalized_hyperbolic(self.amt2, k, self.delay2, s)

        if self.modeltype == 'Q':
            b = params[2]
            util1 = self.__quasi_hyperbolic(self.amt1, k, self.delay1, b)
            util2 = self.__quasi_hyperbolic(self.amt2, k, self.delay2, b)

        dv = util2 - util1

        if onesided_test:
            if np.all(dv < 0) or np.all(dv > 0):
                if len(params) == 2:
                    warnings.warn(f'All predicted choices one-sided with parameter: {math.exp(params[0])}, {math.exp(params[1])}')
                else:
                    warnings.warn(f'All predicted choices one-sided with parameters: {math.exp(params[0])}, {math.exp(params[1])}, {math.exp(params[2])}')
            return None
        
        dv_choice = -np.where(self.choice == 1, dv, -dv)
        reg = np.divide(dv_choice, inverse_temp)
        logp = np.array([-np.log(1 + np.exp(reg[i])) if reg[i] < 709 else -reg[i] for i in range(len(reg))])

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
    def __bounds_e(amt1, delay1, amt2, delay2):
        indifference_ks = np.divide(np.subtract(np.log(amt2), np.log(amt1)), np.subtract(delay2, delay1))
        return min(indifference_ks), max(indifference_ks)
    

    @staticmethod
    def __bounds_h(amt1, delay1, amt2, delay2):
        indifference_ks = np.divide(np.subtract(amt2, amt1), np.subtract(np.multiply(amt1, delay2), np.multiply(amt2, delay1)))
        return min(indifference_ks), max(indifference_ks)


    @staticmethod
    def __itc_input_checker(modeltype, choice, amt1, delay1, amt2, delay2):

        modeltypes = ['E', 'H', 'GH', 'Q']

        assert (type(modeltype) == str and modeltype.upper() in modeltypes), f'{modeltype} should be a string from the list "E" (exponential), "H" (hyperbolic), "GH" (generalized hyperbolic), and "Q" (quasi hyperbolic)'
        modeltype = modeltype.upper()

        arraylike_inputs = [choice, amt1, delay1, amt2, delay2]
        arraylike_labels = ['choice', 'amt1', 'delay1', 'amt2', 'delay2']

        for i in range(len(arraylike_inputs)):

            if not isinstance(arraylike_inputs[i], np.ndarray):
                try:
                    arraylike_inputs[i] = np.array(arraylike_inputs[i])
                except Exception as e:
                    raise RuntimeError(f'{arraylike_labels[i]} should be an array-like; error converting to numpy array: {e}')
            
            try:
                arraylike_inputs[i] = arraylike_inputs[i].astype(float)
            except Exception as e:
                raise RuntimeError(f'{arraylike_labels[i]} should only contain numerical values, error casting to float: {e}')
            
            assert (type(arraylike_inputs[i]) == np.ndarray and arraylike_inputs[i].ndim == 1), f'{arraylike_labels[i]} should be a vector'
            assert (arraylike_inputs[i].size > 2), f'{arraylike_labels[i]} should have at least 3 elements'

            if i == 0:
                assert (np.all((arraylike_inputs[i] == 0) | (arraylike_inputs[i] == 1))), f'all elements in {arraylike_labels[i]} should be 1 or 0'
            else:
                assert (np.all(arraylike_inputs[i] >= 0)), f'{arraylike_labels[i]} should be nonnegative numbers only'

        assert (choice.size == amt1.size == delay1.size == amt2.size == delay2.size), 'all vectors should have equal size'

        return modeltype, arraylike_inputs[0], arraylike_inputs[1], arraylike_inputs[2], arraylike_inputs[3], arraylike_inputs[4]
