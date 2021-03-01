# computational imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import json
# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


def knapsack_local(weights,values):
    max_iter = 1000
    max_weight = 50
    np.random.seed()
    num_improvements = 0
    num_items = len(weights)
    x = np.zeros(num_items, dtype = bool)
    tot_weight = sum( weights[x] )
    tot_value = sum( values[x] )
    for i in range(max_iter):
        bit_to_flip = np.random.randint(num_items)
        x[bit_to_flip] = ~x[bit_to_flip]
        weight = sum( weights[x] )
        value = sum( values[x] )
        if(weight <= max_weight and value > tot_value):
            tot_weight = weight
            tot_value = value
            num_improvements += 1
        else:
            # No improvement so flip bit back
            x[bit_to_flip] = ~x[bit_to_flip]
    print(f"num_improvement={num_improvements}")
    return x, tot_value, tot_weight

num_items = 20
np.random.seed(seed=123)
values = np.random.randint(low=5, high=50, size=num_items)
weights = np.random.randint(low=1, high=10, size=num_items)
print(values)
print(weights)

items, value, weight = knapsack_local(weights, values)
print(f"items: {items}")
print(f"value: {value}")
print(f"weight: {weight}")


