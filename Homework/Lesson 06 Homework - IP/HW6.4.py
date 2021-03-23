


#Problem Data - generate random weights and values for a knapsack problem (do not change)
import numpy as np
num_items = 20

np.random.seed(seed=123)

values = np.random.randint(low=5, high=50, size=num_items)
weights = np.random.randint(low=1, high=10, size=num_items)
np.random.seed() # use system clock to reset the seed so future random numbers will appear random


max_weight = 50
items = [i for i in range(num_items)]

from pyomo.environ import *

# Instantiate concrete model
M = ConcreteModel(name="HW6.4")

M.y = Var(items,domain=Boolean) # <- 0's and 1's

# maximize(v1 * y1 + v2 * y2 + ...)

# subject to
# total weight <= 50
M.y.pprint()


# Objective:  Maximize Profit
M.value = Objective(expr=sum(values[i]*M.y[i] for i in items),
                     sense=maximize)
M.value.pprint()

# Constraints:
M.constraints = ConstraintList()

# choose only 3 routes
M.constraints.add(sum(weights[i]*M.y[i] for i in items) <= max_weight)


M.constraints.pprint()

# Solve
solver = SolverFactory('glpk')
solver.solve(M)

print(f"\nMaximum value is {M.value()}")
print(f"\nWeight is {sum(weights[i]*M.y[i].value for i in items)}")

print(f"\nWhich items to put in bag:{[i for i in items if M.y[i].value==1]}")
