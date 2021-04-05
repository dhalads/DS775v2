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

# solution printer

class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self):
        self.__solution_count += 1
        for v in self.__variables:
            print(f'{v} = {self.Value(v)}', end = ' ')
        print()

    def solution_count(self):
        return self.__solution_count

# abstract assignment problem

# problem data
locations = ['Loc1', 'Loc2', 'Loc3', 'Loc4']
machines = ['Mach1', 'Mach2', 'Mach3']
cost_table = [[13, 16, 12, 11], [15, 99, 13, 20], [5, 7, 10, 6]]

num_locations = len(cost_table[0])
num_machines = len(cost_table)

# Create the model.
model = cp_model.CpModel()

# Variables
assign = [
    model.NewIntVar(0, num_locations - 1, machines[i])
    for i in range(num_machines)
]

#get the maximum cost from the cost_table for the top of our intvar range
max_cost = max(list(map(max, cost_table)))
cost = [model.NewIntVar(0, max_cost, f'cost{i}') for i in range(num_machines)]

# Constraints
model.AddAllDifferent(assign)

for i in range(num_machines):
    model.AddElement(assign[i], cost_table[i], cost[i]) #cost[i]= cost_table[i][assign[i]] 

model.Minimize(sum(cost))

# Creates a solver and solves the model.
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL:
    print(f'Lowest Possible Cost: {solver.ObjectiveValue()}')
    print()
    print('Assignments and associated costs:')
    cost_assigns = pd.DataFrame(0, index=machines, columns=locations)
    for i in range(num_machines):
        cost_assigns.iloc[i, solver.Value(assign[i])] = solver.Value(cost[i])
    display(cost_assigns)