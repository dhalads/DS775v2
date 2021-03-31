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

course = ['1', '2', '3', '4']

study_days = [1, 2, 3, 4]
gps = [ [3,5,6,7], [5,5,6,9], [2,4,7,8], [6,7,9,9] ]
# cost_per_unit = {
#     plants[p]: dict(zip(dist, cpu[p][:]))
#     for p in range(len(plants))
# }

num_courses = len(course)
num_study_days = len(study_days)
# Create the model.
model = cp_model.CpModel()

# Variables
assign = [
    model.NewIntVarFromDomain(cp_model.Domain.FromValues(study_days), f'x{c}')
    for c in range(num_courses)
]

#get the maximum cost from the cost_table for the top of our intvar range
max_pts = max(list(map(max, gps)))
points = [model.NewIntVar(0, max_pts, f'points{c}') for c in range(num_courses)]

# Constraints
# model.AddAllDifferent(assign)

for c in range(num_courses):
    model.AddElement(assign[c], gps[c], points[c]) #cost[i]= cost_table[i][assign[i]]

model.Maximize(sum(points))

model.Add(sum(assign[c] for c in range(num_courses))==7)

for c in range(num_courses):
    model.Add(assign[c] >= 1)

# Creates a solver and solves the model.
solver = cp_model.CpSolver()
status = solver.Solve(model)
# tmp = cp_model.INFEASIBLE
if status == cp_model.OPTIMAL:
    print(f'Lowest Possible Cost: {solver.ObjectiveValue()}')
    print()
    print('Assignments and associated costs:')
    cost_assigns = pd.DataFrame(0, index=course, columns=study_days)
    for i in range(num_courses):
        cost_assigns.iloc[i, solver.Value(assign[i])] = solver.Value(points[i])
    display(cost_assigns)
else:
    print(f"status={status}")



