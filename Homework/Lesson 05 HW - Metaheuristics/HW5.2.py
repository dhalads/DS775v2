# generate random weights and values for a knapsack problem
# DO NOT CHANGE ANYTHING in this block of code
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from scipy.optimize import minimize
import json
from simanneal import Annealer



def create_set1():
    num_items = 20
    np.random.seed(seed=123)
    values = np.random.randint(low=5, high=50, size=num_items)
    weights = np.random.randint(low=1, high=10, size=num_items)
    x = np.zeros(num_items, dtype = bool)  # all false
    return values, weights, x

def print_set(label):
    weight = sum( weights[x] )
    value = sum( values[x] )
    print(label)
    print(f"items selected: {x}")
    print(f"values: {values}")
    print(f"weights: {weights}")



def print_out(label):
    weight = sum( weights[x] )
    value = sum( values[x] )
    print(label)
    print(f"items selected: {x}")
    print(f"value: {value}")
    print(f"weight: {weight}")





class KnapsackProblem_a(Annealer):

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, values, weights, max_weight):
        self.values = values
        self.weights = weights
        self.max_weight = max_weight
        super(KnapsackProblem_a, self).__init__(state)  # important!

    def move(self):
        new_state = self.state.copy()
        bit_to_flip = np.random.randint(new_state.size)
        new_state[bit_to_flip] = ~new_state[bit_to_flip]
        weight = sum( self.weights[new_state] )
        if(weight <= self.max_weight):
            self.state = new_state

    def energy(self):
        return -sum( self.values[self.state] )

max_weight = 50
np.random.seed() # use system clock to reset the seed so future random numbers will appear random

# values, weights, x = create_set1()
# x[[(0,2,4)]] = True
# print_set("prob a")
# ksp = KnapsackProblem_a(x, values, weights, max_weight)
# ksp.set_schedule(ksp.auto(minutes=.2)) #set approximate time to find results

# x, value = ksp.anneal()
# print_out(f"prob a")

class KnapsackProblem_b(Annealer):

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, values, weights, max_weight):
        self.values = values
        self.weights = weights
        self.max_weight = max_weight
        super(KnapsackProblem_b, self).__init__(state)  # important!

    def move(self):
        new_state = self.state.copy()
        bit_to_flip = np.random.randint(new_state.size)
        new_state[bit_to_flip] = ~new_state[bit_to_flip]
        self.state = new_state

    def energy(self):
        tot_value = sum(self.values[self.state])
        penalty = sum(self.values)*min( self.max_weight - sum(self.weights[self.state]), 0)
        return -(tot_value+penalty)

values, weights, x = create_set1()
x[[(0,2,4)]] = True
print_set("prob b")
ksp = KnapsackProblem_b(x, values, weights, max_weight)
ksp.set_schedule(ksp.auto(minutes=.2)) #set approximate time to find results

x, value = ksp.anneal()
print_out(f"prob b")
