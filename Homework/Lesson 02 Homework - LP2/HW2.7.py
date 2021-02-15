# data for problem 7.3-7
periods = [
    'p6_8', 'p8_10', 'p10_12', 'p12_14', 'p14_16', 'p16_18', 'p18_20',
    'p20_22', 'p22_24', 'p24_6'
]
shifts = ['s1', 's2', 's3', 's4', 's5']
daily_cost_per_agent = dict( zip( shifts, [170, 160, 175, 180, 195] ) )
min_agents_per_period = dict( zip( periods, [48, 79, 65, 87, 64, 73, 82, 50, 52, 15] ) )
                            
pc = [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0],
     [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 0],
     [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]]
periods_covered = { periods[p]: dict(zip( shifts, pc[p][:])) for p in range(len(periods))}

# finish the code here:

from pyomo.environ import *

model = ConcreteModel()

model.agents_each_shift = Var(shifts, domain=NonNegativeReals)

model.total_cost = Objective(expr=sum(daily_cost_per_agent[s] * model.agents_each_shift[s]
                                      for s in shifts),
                             sense=minimize)

model.total_cost.pprint()

model.con_min_agents_each_period = ConstraintList()
for p in periods:
    model.con_min_agents_each_period.add(
        sum(model.agents_each_shift[s] * periods_covered[p][s] for s in shifts) >= min_agents_per_period[p])

model.con_min_agents_each_period.pprint()


# solve and display
solver = SolverFactory('glpk')
solver.solve(model)

# display solution
print(f"Minimum Total Cost = ${model.total_cost():,.2f}")
model.agents_each_shift.pprint()

model.write('model.lp', io_options={'symbolic_solver_labels': True})
#!glpsol -m model.lp --lp --ranges sensit.sen
command = "glpsol -m model.lp --lp --ranges sensit.sen"
import subprocess
subprocess.call(command)

import numpy as np
# print sensitivity report
np.set_printoptions(linewidth=110)
f = open('sensit.sen', 'r')
file_contents = f.read()
print(file_contents)
f.close()
