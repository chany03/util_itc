A package for fitting intertemporal choice data to models. 

Available models for fitting:
* Exponential: $U=A·exp(−kD)$
* Hyperbolic: $U=A·(1+kD)^{−1}$
* Generalized Hyperbolic: $U=A·(1+kD)^{−s}$
* Quasi-hyperbolic: $U=A·βexp(−kD)$

The constructor takes a model type and data (choices, payoffs, and delays) for two options and instantiates a util-itc object, fitting the data during instantiation. 

The resulting object stores the fitted parameters (k, inverse temperature, and an extra parameter s or b for generalized hyperbolic or quasi-hyperbolic models, respectively) in an instance variable, output.

Warnings will be issued if all choices in the input data are one-sided (all 0 or 1), or if the fitted model predicts all one-sided choices.

Dependencies: numpy version >= 1.26.4, scipy version >= 1.12.0

To install using pip and import the package, copy the following code into your terminal:
```
pip install util-itc
from util_itc import util_itc
```

To fit after importing, construct a util_itc object for each set of data you would like to fit, in the following format:
```
x = util_itc(modeltype, choice, amt1, delay1, amt2, delay2)
```
where x is the variable that results will be stored in.
Modeltype should be a 1-length string ('E', 'H', 'GH', or 'Q') that will determine the model used for fitting. All other parameters should be arraylike objects (numpy arrays, lists, etc.).

To obtain fitted parameters, view the output instance variable:
```
y = x.output
print(y)
```
where y will store the fitted results in the following format:
[[k, inverse temperature, optional parameter s/b], 'modeltype', number of data points]

Example of use:
```
>>> example = util_itc("E", [0, 1, 0], [9, 9, 9], [1, 4, 3], [1, 7, 4], [4, 0, 0])
>>> example.output
[[0.1428602660107752, 0.36787944117144233], 'E', 3]
```

For queries regarding package maintenance, please contact chanyoungchung@berkeley.edu

To view additional information about the class or functions, use python's help function (with package installed):
```
help(util_itc)
help(util_itc.fun)
```
To exit the help window in terminal, press q.

Help function documentation:

```
class util_itc(builtins.object)
 |  util_itc(modeltype, choice, amt1, delay1, amt2, delay2)
 |  
 |  Takes intertemporal choice data for n >= 3 decisions and returns estimated parameters k, inverse temperature, and an extra parameter where relevant for the model.
 |  
 |  Args:
 |  modeltype: string describing the model used to fit data: 'E' for exponential, 'H' for hyperbolic, 'GH' for generalized hyperbolic, or 'Q' for quasi-hyperbolic.
 |  choice: array-like of size n containing only the values 1 and 0, where 1 represents option 1 in the choice data and 0 represents option 2.
 |  amt1: array-like of size n containing nonnegative numbers, where each value represents a payoff from option 1
 |  delay1: array-like of size n containing nonnegative numbers, where each value represents a delay before receiving a payoff from option 1
 |  amt2: array-like of size n containing nonnegative numbers, where each value represents a payoff from option 2
 |  delay2: array-like of size n containing nonnegative numbers, where each value represents a delay before receiving a payoff from option 2
 |  
 |  Validates inputs, then runs the fit method to fit intertemporal choice data to a model of the type given.
 |  Stores parameters in instance variable named output, formatted as: [[est. k, est. inverse temperature, est. extra parameter (s for GH, b for Q)], "model", number of choices]
 |  
 |  Methods defined here:
 |  
 |  __init__(self, modeltype, choice, amt1, delay1, amt2, delay2)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  calculate_dv(self, params)
 |  
 |  fit(self)
 |      Uses scipy.optimize.minimize to fit intertemporal choice data to a model specified during object initialization.
 |  
 |  fun(self, params)
 |      Defines the objective function to be minimized, calculating utility differences based on model type and given parameters.
 |      
 |      Args:
 |      params: a size 2 or 3 list containing initial parameter starting points for k, inverse temperature, and an optional second parameter s or b.
 |      
 |      Returns:
 |      float: negative average log likelihood of choices
 ```