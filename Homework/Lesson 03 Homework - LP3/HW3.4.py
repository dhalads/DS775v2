from pyomo.environ import *
import pandas as pd

swimmers = ['Carl', 'Chris', 'David', 'Tony', 'Ken']
supply = dict(zip(swimmers, [1, 1, 1, 1, 1]))


strokes = ['back', 'beast', 'butterfly', 'freestyle', 'dummy']
demand = dict(zip(strokes, [1, 1, 1, 1, 1]))

bigM = 1000
time_list = [[37.7,43.4,33.3,29.2,bigM],[32.9,33.1,28.5,26.4,bigM],[33.8,42.2,38.9,29.6,bigM],[37.0,34.7,30.4,28.5,bigM],[35.4,41.8,33.6,31.1,bigM]]
time = {
    swimmers[sw]: {strokes[st]: time_list[sw][st]
                   for st in range(len(strokes))}
    for sw in range(len(swimmers))
}

model = ConcreteModel()

model.assign= Var(swimmers, strokes, domain=NonNegativeReals)

model.total_time = Objective(expr=sum(time[sw][st] * model.assign[sw, st]
                                      for sw in swimmers for st in strokes if st != 'dummy'),
                             sense=minimize)

# model.total_time.pprint()

model.supply_ct = ConstraintList()
for sw in swimmers:
    model.supply_ct.add(
        sum(model.assign[sw, st] for st in strokes) == supply[sw])


model.demand_ct = ConstraintList()
for st in strokes:
    model.demand_ct.add(
        sum(model.assign[sw, st] for sw in swimmers) == demand[st])



model.value_ct = ConstraintList()
for sw in swimmers:
    for st in strokes:
        model.value_ct.add(model.assign[sw, st]  <= 1)

# solve and display
solver = SolverFactory('glpk')
solver.solve(model)

# display solution
print(f"Minimum total time in seconds = {model.total_time():,.1f}")

# put amounts in dataframe for nicer display
dvars = pd.DataFrame([[model.assign[sw, st]() for sw in swimmers]
                      for st in strokes],
                     index = strokes,
                     columns=swimmers)
print("Swimmer assignments to strokes:")
print(dvars)