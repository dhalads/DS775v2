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


# generalizable schedule code

task_duration_dict = {
    'A': 2,
    'B': 4,
    'C': 10,
    'D': 6,
    'E': 4,
    'F': 5,
    'G': 7,
    'H': 9,
    'I': 7,
    'J': 8,
    'K': 4,
    'L': 5,
    'M': 2,
    'N': 6
}
task_names = list(task_duration_dict.keys())
num_tasks = len(task_names)
durations = list(task_duration_dict.values())

# for each task we have a list of tasks that must go after
# task:['these','tasks','after']
precedence_dict = {
    'B': ['A'],
    'C': ['B'],
    'D': ['C'],
    'E': ['C'],
    'F': ['E'],
    'G': ['D'],
    'H': ['E','G'],
    'I': ['C'],
    'J': ['F', 'I'],
    'K': ['J'],
    'L': ['J'],
    'M': ['H'],
    'N': ['K', 'L']
}

task_name_to_number_dict = dict(zip(task_names, np.arange(0, num_tasks)))

horizon = sum(task_duration_dict.values())

model = cp_model.CpModel()

start_vars = [
    model.NewIntVar(0, horizon, name=f'start_{t}') for t in task_names
]
end_vars = [model.NewIntVar(0, horizon, name=f'end_{t}') for t in task_names]

# the `NewIntervalVar` are both variables and constraints, the internally enforce that start + duration = end
intervals = [
    model.NewIntervalVar(start_vars[i],
                         durations[i],
                         end_vars[i],
                         name=f'interval_{task_names[i]}')
    for i in range(num_tasks)
]

# precedence constraints
for after in list(precedence_dict.keys()):
    for before in precedence_dict[after]:
        before_index = task_name_to_number_dict[before]
        after_index = task_name_to_number_dict[after]
        model.Add(end_vars[before_index] <= start_vars[after_index])

obj_var = model.NewIntVar(0, horizon, 'largest_end_time')
model.AddMaxEquality(obj_var, end_vars)
model.Minimize(obj_var)

solver = cp_model.CpSolver()
status = solver.Solve(model)

print(f'Optimal Schedule Length: {solver.ObjectiveValue()}')
for i in range(num_tasks):
    print(
        f'{task_names[i]} start at {solver.Value(start_vars[i])} and end at {solver.Value(end_vars[i])}'
    )

# output_notebook()

starts = [solver.Value(start_vars[i]) for i in range(num_tasks)]
ends = [solver.Value(end_vars[i]) for i in range(num_tasks)]

source = ColumnDataSource(data=dict(tasks=task_names, starts = starts, ends=ends))

p = figure(x_range=(0,solver.ObjectiveValue()), y_range=task_names, plot_height=350, title="Task Time Spans",
           toolbar_location=None, tools="")

p.hbar(y='tasks', left='starts', right='ends', height=0.9, source=source)

p.xaxis.axis_label = "Time"
p.ygrid.grid_line_color = None

show(p)