import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from scipy.optimize import minimize
import json
from simanneal import Annealer

# Self-Assessment Solution for Simulated Annealing with Continuous Variables

# rastrigin definition
def f(x):
    x = np.array(x) # force a numpy arrray here so that the math below works
    # pass a single vector of length n (=dim) to evaluate Rastrigin
    return sum(x**2 + 10 - 10 * np.cos(2 * np.pi * x))

def gauss_move(xy,sigma):
    # xy is a 1 by dim numpy array
    # sigma is the standard deviation for the normal distribution
    dim = len(xy)
    return xy + np.random.normal(loc = 0, scale = sigma, size=dim)

def clip_to_bounds(xy,low,high):
    # xy is a 1 by dim numpy array
    # low is the lower bound for clipping variables
    # high is the upper bound for clipping variables
    return np.array( [min(high,max(low,v)) for v in xy])

class Rastrigin_a(Annealer):

    # no extra data so just initialize with state
    def __init__(self, state, sigma, low, high):
        self.sigma = sigma
        self.low = low
        self.high = high
        super(Rastrigin_a, self).__init__(state)  # important!

    def move(self):
        self.state = gauss_move(self.state, self.sigma)
        self.state = clip_to_bounds(self.state, self.low, self.high)

    def energy(self):
        return f(self.state)



low = -5.12
high = 5.12
sigma = (high-low)/6
# init_state = np.random.uniform(low=low,high=high,size=10)
# init_state = np.random.uniform(low=0,high=0,size=10)
# problem2D = Rastrigin_a( init_state, sigma, low, high )
# problem2D.set_schedule(problem2D.auto(minutes=.6))
# best_x, best_fun = problem2D.anneal()

# print("Notice that the results below are displayed using scientific notation.\n")
# print(f"The lowest function value found by simulated annealing is {best_fun:.3e}")
# print(f"That value is achieved when x = {best_x}")
# # refine with local search
# from scipy.optimize import minimize

# result = minimize(f,best_x)
# print("\nAfter refining the result from simulated annealing with local search.")
# print(f"The lowest function value found by local search is {result.fun:.3e}")
# print(f"That value is achieved when x = {result.x[0]:.3e} and y = {result.x[1]:.3e}")

class Rastrigin_b(Annealer):

    # no extra data so just initialize with state
    def __init__(self, state, sigma, low, high):
        self.sigma = sigma
        self.low = low
        self.high = high
        super(Rastrigin_b, self).__init__(state)  # important!

    def move(self):
        self.state = gauss_move(self.state, self.sigma)
        self.state = clip_to_bounds(self.state, self.low, self.high)
        bounds = [(self.low,self.high) for i in range(self.state.size)]
        result = minimize(f, self.state, bounds=bounds, tol=0.1)
        self.state = result.x

    def energy(self):
        return f(self.state)

init_state = np.random.uniform(low=low,high=high,size=10)
problem2D = Rastrigin_b( init_state, sigma, low, high )
problem2D.set_schedule(problem2D.auto(minutes=.001))
best_x, best_fun = problem2D.anneal()
bounds = [(low,high) for i in range(best_x.size)]
result = minimize(f, best_x, bounds=bounds)

print("Notice that the results below are displayed using scientific notation.\n")
print(f"The lowest function value found by simulated annealing is {best_fun:.3e}")
print(f"That value is achieved when x = {best_x}")