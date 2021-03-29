# EXECUTE FIRST

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



# Create the model.
model = cp_model.CpModel()
x_sets = [[3,6,12],[3,6],[3,6,9,12],[6,12],[9,12,15,18]]
x = [model.NewIntVarFromDomain(cp_model.Domain.FromValues(x_sets[i]), f'x{i}')  for i in range(len(x_sets))]
xsq = [model.NewIntVar(0,2500, f'x{i}sq')  for i in range(len(x_sets))]

model.AddAllDifferent([ x[i] for i in range(len(x))])
for i in range(len(x)):
    model.AddMultiplicationEquality(xsq[i], [x[i], x[i]])
# Creates the constraints.
model.Add(x[1-1] + x[3-1] + x[4-1] <= 25)

# Creates a solver and solves the model.
solver = cp_model.CpSolver()
solution_printer = VarArraySolutionPrinter([x[i] for i in range(len(x_sets))])
status = solver.SearchForAllSolutions(model, solution_printer)

print(f'Status = {solver.StatusName(status)}')
print(f'Number of solutions found: {solution_printer.solution_count()}')


# Add an objective function and a direction, need not be linear
model.Maximize(5*x[1-1] - xsq[1-1] + 8*x[2-1] - xsq[2-1] + 10*x[3-1] - xsq[3-1] + 15*x[4-1] - xsq[4-1] + 20*x[5-1] - xsq[5-1])


# Creates a solver and solves the model.
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL:
    print('Maximum of objective function: %i' % solver.ObjectiveValue())
    print()
    for i in range(len(x_sets)):
        print(f"x{i+1} = {solver.Value(x[i])}")



