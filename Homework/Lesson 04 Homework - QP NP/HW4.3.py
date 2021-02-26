# computational imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import json
# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

# plot p(x) on [-10,10]
x = np.linspace(0,5,201)
f = lambda x:100*x**6 - 1359*x**5 + 6836*x**4 - 15670*x**3 + 15870*x**2 - 5095*x
fig = plt.figure(figsize=(8,7)) # adjust figsize if needed
plt.plot(x,f(x));
plt.xlabel('x');
plt.ylabel('y');

plt.show(block=True)


from scipy.optimize import minimize

# find minima first
x0_min = [.1, 2, 4.5]
for x0 in x0_min:
    result = minimize( f, x0, bounds = [(0,5)])
    print(f"There is a local minimum value of {result.fun[0]:.2f} at x = {result.x[0]:.2f}")

# now maxima
neg_f = lambda x:-f(x)
x0_max = [.1,1.1,3.1,4.9]
for x0 in x0_max:
    result = minimize( neg_f, x0, bounds = [(0,5)])
    print(f"There is a local maximum value of {-result.fun[0]:3.2f} at x = {result.x[0]:1.2f}")


print('\n')

def multistart_run(func, lb, ub, dim, num_local_searches):
    bounds = [(lb,ub) for i in range(dim)]
    x_initial = np.random.uniform(lb, ub, dim)
    result = minimize(func, x_initial, bounds=bounds)
    limit_f = result.fun
    limit_x = result.x
    for i in range(num_local_searches):
        x_initial = np.random.uniform(lb, ub, dim)
        result = minimize(func, x_initial, bounds=bounds)
        if(result.fun < limit_f):
            limit_f = result.fun
            limit_x = result.x
    return limit_f, limit_x

def neg_f(x):
    return -1 * f(x)

dim = 1
num_local_searches = 100
limit_f, limit_x = multistart_run(neg_f, 0, 5, dim,num_local_searches)
print(f" global max is {-limit_f} at x={limit_x}")

