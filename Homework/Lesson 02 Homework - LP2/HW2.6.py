lab_days = ['Mon', 'Tue', 'Wed', 'Thurs', 'Fri']
max_hours_worked_per_day = dict(zip(lab_days, [14, 14, 14, 14, 14]))

operators = ['KC', 'DH', 'HB', 'SC', 'KS', 'NK']
operator_wage = dict(zip(operators, [25, 26, 24, 23, 28, 30]))
operator_min_hours_per_week = dict(zip(operators, [8, 8, 8, 8, 7, 7]))

omhed = [[6,0,6,0,6], [0,6,0,6,0], [4,8,4,0,4], [5,5,5,0,5], [3,0,3,8,0], [0,0,0,6,2]]
operator_max_hours_each_day = {
    operators[o]: {lab_days[d]: omhed[o][d]
                   for d in range(len(lab_days))}
    for o in range(len(operators))
}

# throw an error if total supply and demand do not match
# assert (sum(supply.values()) == sum(demand.values()))

from pyomo.environ import *

model = ConcreteModel()

model.operator_hours = Var(operators, lab_days, domain=NonNegativeReals)

model.total_cost = Objective(expr=sum(operator_wage[o] * model.operator_hours[o, d]
                                      for o in operators for d in lab_days),
                             sense=minimize)

model.total_cost.pprint()

model.min_proficiency_hours = ConstraintList()
for o in operators:
    model.min_proficiency_hours.add(
        sum(model.operator_hours[o, d] for d in lab_days) >= operator_min_hours_per_week[o])

model.min_proficiency_hours.pprint()

model.max_total_hours_each_day = ConstraintList()
for d in lab_days:
    model.max_total_hours_each_day.add(
        sum(model.operator_hours[o, d] for o in operators) == max_hours_worked_per_day[d])

model.max_total_hours_each_day.pprint()

model.con_operator_max_hours_each_day = ConstraintList()
for o in operators:
    for d in lab_days:
        model.con_operator_max_hours_each_day.add(
            model.operator_hours[o, d]  <= operator_max_hours_each_day[o][d])

model.con_operator_max_hours_each_day.pprint()

# solve and display
solver = SolverFactory('glpk')
solver.solve(model)

# display solution
print(f"Minimum Total Cost = ${model.total_cost():,.2f}")

# put amounts in dataframe for nicer display
import pandas as pd
dvars = pd.DataFrame([[model.operator_hours[o, d]() for d in lab_days]
                      for o in operators],
                     index=operators,
                     columns=lab_days)
print("Number of hours each person will work each day:")
print(dvars)

model.write('model.lp', io_options={'symbolic_solver_labels': True})
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