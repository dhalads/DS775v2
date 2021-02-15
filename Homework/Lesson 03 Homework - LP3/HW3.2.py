# using openpyxl
from openpyxl import load_workbook
from pyomo.environ import *
import pandas as pd
import numpy as np
import os
print(os.getcwd())


sources = [1, 2, 3]
markets = [1, 2, 3, 4, 5]
modes = ['rail', 'ship']
supply_dict = dict( zip( sources, [15, 20, 15] ) )
demand_dict = dict( zip( markets, [11, 12, 9, 10, 8] ) )

bigM = 10000

def add_costs(s, m, rate, invest):
    key_list = []
    for mk in markets:
        key_list.append((s,mk,m))
    values = list(zip(rate, invest))
    cost_dict = dict( zip( key_list, values ) )
    costs.update(cost_dict)

def run_model(cmod):
    # remove bigM items
    keys = [k for k, v in cmod.items() if v[0] == bigM]
    for x in keys:
        del cmod[x]

    model = ConcreteModel()
    model.num_shipped_per_route = Var(cmod.keys(), domain = NonNegativeReals)

    # define objective function
    model.total_cost = Objective( expr = sum(cmod[s, mk, mode][0]*model.num_shipped_per_route[s, mk, mode] + cmod[s, mk, mode][1]
                                             for (s, mk, mode) in cmod.keys() ),
                                sense = minimize )
    model.total_cost.pprint()

    # define constraints
    model.supply_ct = ConstraintList()

    for s in sources:
        model.supply_ct.add(
            sum(model.num_shipped_per_route[ls, mk, mode] for ls, mk, mode in costs.keys() if s == ls ) == supply_dict[s])

    model.supply_ct.pprint()

    model.demand_ct = ConstraintList()
    for (mk) in demand_dict.keys():
        model.demand_ct.add(
            sum(model.num_shipped_per_route[s, mkl, mode] for s, mkl, mode in costs.keys() if mk == mkl ) == demand_dict[mk])

    model.demand_ct.pprint()

    solver = SolverFactory('glpk')
    solver.solve(model)

    rail = pd.DataFrame(0, index=sources, columns=markets)
    for (s, mk, mode) in costs:
        if ( mode == 'rail'):
            rail.loc[s, mk] = model.num_shipped_per_route[s, mk, mode].value

    ship = pd.DataFrame(0, index=sources, columns=markets)
    for (s, mk, mode) in costs:
        if ( mode == 'ship'):
            ship.loc[s, mk] = model.num_shipped_per_route[s, mk, mode].value

    output = (model.total_cost(), rail, ship)
    return output

def print_output(label, input):
    print(f"\nTransported Amounts for {label}")
    print(f"Minimum Total Cost = ${input[0]:,.2f}")

    print(f"\nTransported Amounts for rail")
    print(input[1])
    print(f"\nTransported Amounts for ship")
    print(input[2])

costs = {}
add_costs(1, 'rail', [61,72,45,55,66], [0,0,0,0,0] )
add_costs(2, 'rail', [69,78,60,49,56], [0,0,0,0,0] )
add_costs(3, 'rail', [59,66,63,61,47], [0,0,0,0,0] )

output = run_model(costs)
print_output("only rail", output)

costs = {}
add_costs(1, 'rail', [61,72,45,55,66], [0,0,0,0,0] )
add_costs(2, 'rail', [69,78,60,49,56], [0,0,0,0,0] )
add_costs(3, 'rail', [59,66,63,61,47], [0,0,0,0,0] )

add_costs(1, 'ship', [31,38,24,bigM,35], np.divide([275,303,238,bigM*10,285],10) )
add_costs(2, 'ship', [36,43,28,24,31], np.divide([293,318,270,250,265],10) )
add_costs(3, 'ship', [bigM,33,36,32,26], np.divide([bigM*10,283,275,268,240],10) )

output = run_model(costs)
print_output("only rail", output)
