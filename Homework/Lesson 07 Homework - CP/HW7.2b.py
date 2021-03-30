# num_vars = 5
# var_names = [f'x{i}' for i in range(1,num_vars+1)]
# var_names

# from ortools.sat.python import cp_model
# model = cp_model.CpModel()
# var_values = {'x1':[3,6,12],
#        'x2':[3,6],
#        'x3':[3,6,9,12],
#        'x4':[6,12],
#        'x5':[9,12,15,18]}
# x = [model.NewIntVarFromDomain(cp_model.Domain.FromValues(var_values[v]),v) for v in var_names]
# x

# computational imports
from ortools.sat.python import cp_model
import numpy as np
import pandas as pd
from bokeh.io import show, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import json
# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
# for reading files from urls
import urllib.request
# display imports
from IPython.display import display, IFrame
from IPython.core.display import HTML

# abstract assignment problem

# problem data

dist = ['1', '2', '3', '4']
units_dist = dict(zip(dist, [20,20,20]))
# product_startup_cost = dict(zip(products, [50000,40000,70000,60000]))
plants = ['A1', 'A2', 'B1', 'B2']
shipments = ['1', '2']
cpu = [ [600, 700, 400,0], [600, 700, 400,0], [700, 800, 500,0], [700, 800, 500,0] ]
cost_per_unit = {
    plants[p]: dict(zip(dist, cpu[p][:]))
    for p in range(len(plants))
}

num_plants = len(plants)
num_shipments = len(shipments)
num_dist = len(dist)
# Create the model.
model = cp_model.CpModel()

# Variables
assign = [
    model.NewIntVar(0, num_dist - 1, plants[p])
    for p in range(num_plants)
]

#get the maximum cost from the cost_table for the top of our intvar range
max_cost = max(list(map(max, cpu)))
cost = [model.NewIntVar(0, max_cost, f'cost{p}') for p in range(num_plants)]

# Constraints
model.AddAllDifferent(assign)

for p in range(num_plants):
    model.AddElement(assign[p], cpu[p], cost[p]) #cost[i]= cost_table[i][assign[i]]

model.Minimize(sum(cost))

# Creates a solver and solves the model.
solver = cp_model.CpSolver()
status = solver.Solve(model)
# tmp = cp_model.INFEASIBLE
if status == cp_model.OPTIMAL:
    print(f'Lowest Possible Cost: {solver.ObjectiveValue()}')
    print()
    print('Assignments and associated costs:')
    cost_assigns = pd.DataFrame(0, index=plants, columns=dist)
    for i in range(num_plants):
        cost_assigns.iloc[i, solver.Value(assign[i])] = solver.Value(cost[i])
    display(cost_assigns)



