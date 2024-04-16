from ITCcode import util_itc
import numpy as np
from random import random, randint
import math
import matplotlib.pyplot as plt


def trial_generator(k_min, k_max, n):
    
    amt1s = []
    delay1s = []
    amt2s = []
    delay2s = []

    for indifference_k in np.arange(k_min + k_max / n, k_max + k_max / n, k_max / n):

        amt1 = 20
        amt1s.append(amt1)

        delay1 = 0
        delay1s.append(delay1)

        amt2 = randint(21, 100)
        amt2s.append(amt2)

        delay2 = round(math.log(amt1 / amt2) / -indifference_k, 2)
        delay2s.append(delay2)

    return np.array(amt1s), np.array(delay1s), np.array(amt2s), np.array(delay2s)


def exponential(a1s, d1s, a2s, d2s):

    random_k = round(random() * 2, 2)
    random_kn = -1 * random_k
    random_it = round(random() * 3, 2)

    util_diffs = np.subtract(np.multiply(a2s, np.exp(np.multiply(random_kn, d2s))), np.multiply(a1s, np.exp(np.multiply(random_kn, d1s))))
    inv_temped = np.divide(util_diffs, random_it)
    logits = np.divide(1, np.add(1, np.exp(np.multiply(-1, inv_temped))))

    choices = np.array([int(random() <= i) for i in logits])

    return random_k, random_it, choices


def hyperbolic(a1s, d1s, a2s, d2s):

    random_k = round(random() * 2, 2)
    random_it = round(random() * 3, 2)

    util_diffs = np.subtract(np.multiply(a2s, np.divide(1, np.add(1, np.multiply(random_k, d2s)))), np.multiply(a1s, np.divide(1, np.add(1, np.multiply(random_k, d1s)))))
    inv_temped = np.divide(util_diffs, random_it)
    logits = np.divide(1, np.add(1, np.exp(np.multiply(-1, inv_temped))))

    choices = np.array([int(random() <= i) for i in logits])

    return random_k, random_it, choices


def generalized_hyperbolic(a1s, d1s, a2s, d2s):

    random_k = round(random() * 2, 2)
    random_it = round(random() * 3, 2)
    random_s = round(random(), 2)

    util_diffs = np.subtract(np.multiply(a2s, np.divide(1, np.power(np.add(1, np.multiply(random_k, d2s)), random_s))), np.multiply(a1s, np.divide(1, np.power(np.add(1, np.multiply(random_k, d1s)), random_s))))
    inv_temped = np.divide(util_diffs, random_it)
    logits = np.divide(1, np.add(1, np.exp(np.multiply(-1, inv_temped))))

    choices = np.array([int(random() <= i) for i in logits])

    return random_k, random_it, random_s, choices


def quasi_hyperbolic(a1s, d1s, a2s, d2s):

    random_k = round(random() * 2, 2)
    random_kn = -1 * random_k
    random_it = round(random() * 3, 2)
    random_b = round(random(), 2)

    util_diffs = np.subtract(np.multiply(a2s, np.multiply(random_b, np.exp(np.multiply(random_kn, d2s)))), np.multiply(a1s, np.multiply(random_b, np.exp(np.multiply(random_kn, d1s)))))
    inv_temped = np.divide(util_diffs, random_it)
    logits = np.divide(1, np.add(1, np.exp(np.multiply(-1, inv_temped))))

    choices = np.array([int(random() <= i) for i in logits])

    return random_k, random_it, random_b, choices


real_params = []
test_params = []
real_extra = []
test_extra = []

"""
for i in range(9):

    amt1, delay1, amt2, delay2 = trial_generator(0, 2, 1000)

    k, it, choice = exponential(amt1, delay1, amt2, delay2)

    test = util_itc('E', choice, amt1, delay1, amt2, delay2)

    real_params.append(k)
    test_params.append(test.output[0][0])

plt.scatter(real_params, test_params)
plt.xlabel('Real Parameter Values')
plt.ylabel('Estimated Parameter Values')
plt.show()
"""
"""
for i in range(9):

    amt1, delay1, amt2, delay2 = trial_generator(0, 2, 1000)

    k, it, choice = hyperbolic(amt1, delay1, amt2, delay2)

    test = util_itc('H', choice, amt1, delay1, amt2, delay2)

    real_params.append(k)
    test_params.append(test.output[0][0])

plt.scatter(real_params, test_params)
plt.xlabel('Real Parameter Values')
plt.ylabel('Estimated Parameter Values')
plt.show()
"""
"""
for i in range(9):

    amt1, delay1, amt2, delay2 = trial_generator(0, 2, 1000)

    k, it, s, choice = generalized_hyperbolic(amt1, delay1, amt2, delay2)

    test = util_itc('GH', choice, amt1, delay1, amt2, delay2)

    real_params.append(k)
    test_params.append(test.output[0][0])

plt.scatter(real_params, test_params)
plt.xlabel('Real Parameter Values')
plt.ylabel('Estimated Parameter Values')
plt.show()
"""

for i in range(9):

    amt1, delay1, amt2, delay2 = trial_generator(0, 2, 1000)

    k, it, b, choice = quasi_hyperbolic(amt1, delay1, amt2, delay2)

    test = util_itc('Q', choice, amt1, delay1, amt2, delay2)

    real_params.append(k)
    test_params.append(test.output[0][0])

plt.scatter(real_params, test_params)
plt.xlabel('Real Parameter Values')
plt.ylabel('Estimated Parameter Values')
plt.show()

