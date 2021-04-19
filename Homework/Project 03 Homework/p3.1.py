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

def run_schedule():
    # generalizable schedule code

    task_info_dict = {
        'A': ("Evacuate", 7, 14, 21, []),
        'B': ("Lay Foundation", 14, 21, 56, ['A']),
        'C': ("Evacuate", 42, 63, 126, []),
        'D': ("Evacuate", 28, 35, 70, []),
        'E': ("Evacuate", 7, 28, 35, []),
        'F': ("Evacuate", 28, 35, 70, []),
        'G': ("Evacuate", 35, 42, 77, []),
        'H': ("Evacuate", 35, 56, 119, []),
        'I': ("Evacuate", 21, 49, 63, []),
        'J': ("Evacuate", 21, 63, 63, []),
        'K': ("Evacuate", 21, 28, 28, []),
        'L': ("Evacuate", 7, 35, 49, []),
        'M': ("Evacuate", 7, 14, 21, []),
        'N': ("Evacuate", 35, 35, 63, [])
    }

    def get_task_duration():
        output = {}
        for task in task_info_dict.keys():
            task_info = task_info_dict.get(task)
            duration = np.random.triangular(left=task_info[1], mode=task_info[2], right=task_info[3], size=1)
            duration_int = np.rint(duration).astype(int)
            output[task] =  duration_int[0].item()
        return(output)

    task_duration_dict = get_task_duration()

    # task_duration_dict2 = {
    #     'A': 2,
    #     'B': 4,
    #     'C': 10,
    #     'D': 6,
    #     'E': 4,
    #     'F': 5,
    #     'G': 7,
    #     'H': 9,
    #     'I': 7,
    #     'J': 8,
    #     'K': 4,
    #     'L': 5,
    #     'M': 2,
    #     'N': 6
    # }
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

    start_vars = [model.NewIntVar(0, horizon, name=f'start_{t}') for t in task_names]

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

    # # output_notebook()

    # starts = [solver.Value(start_vars[i]) for i in range(num_tasks)]
    # ends = [solver.Value(end_vars[i]) for i in range(num_tasks)]

    # source = ColumnDataSource(data=dict(tasks=task_names, starts = starts, ends=ends))

    # p = figure(x_range=(0,solver.ObjectiveValue()), y_range=task_names, plot_height=350, title="Task Time Spans",
    #         toolbar_location=None, tools="")

    # p.hbar(y='tasks', left='starts', right='ends', height=0.9, source=source)

    # p.xaxis.axis_label = "Time"
    # p.ygrid.grid_line_color = None

    # show(p)
    days = solver.ObjectiveValue()
    profit = 5.4
    if days <= 280:
        profit = profit + 0.150
    elif days > 329 :
        overdays = days - 329
        profit = profit - 0.025*overdays

    output = (days, profit)
    return(output)
    # end method run_schedule

def print_prob_profit(sims, low, high):
    n = len(sims)
    subset = [i for i in sims if i[0]>=low and i[0]<=high]
    prob = len(subset)/n
    if len(subset)==0 :
        avg_profit = 0
    else:
        avg_profit = sum([i[1] for i in subset])/len(subset)
    print(f"Days from {low} to {high}: prob is {prob}, average profit is {avg_profit} ")

def run_simulations(num_sim, isArtifacts):
    days = []
    profits = []
    sims = []
    for i in range(num_sim):
        output = run_schedule()
        days.append(output[0])
        profits.append(output[1])
        sims.append(output)
    print_prob_profit(sims, 0, 280)
    print_prob_profit(sims, 281, 328)
    print_prob_profit(sims, 330, 1000)

run_simulations(10, False)