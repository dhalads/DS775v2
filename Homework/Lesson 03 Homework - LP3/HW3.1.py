# using openpyxl
from openpyxl import load_workbook
from pyomo.environ import *
import pandas as pd
import os
print(os.getcwd())
wb = load_workbook(filename="./data/transp_prob_1.xlsx", data_only=True)
sheet = wb.active

# specify upper left and lower right cells, returns a list or list of lists representing rows
def read_range(sheet, begin, end):
    table = sheet[begin:end]
    height = len(table)
    width = len(table[0])
    if height == 1 or width == 1:
        # for a single row or column produce a list
        tmp = [cell.value for row in table for cell in row]
    else:
        # for an array of cells produces a list of row lists
        tmp = [[cell.value for cell in row] for row in table]
    return (tmp)


# finish reading the data
warehouses = read_range(sheet, 'A3', 'A5')
# stores = ...
wares_stores = [(w,s) for [w,s] in read_range(sheet,'D3','E31')]
# capacity_dict = ...
cost_dict = {(w,s):cost for [w,s,cap,cost] in read_range(sheet,'D3','G31')}
cap_dict = {(w,s):cap for [w,s,cap,cost] in read_range(sheet,'D3','G31')}
supply_dict = { w:q for [w,q] in read_range(sheet,'I3','J5')}
# demand_dict = ...
demand_dict =  { s:d for [s,d] in read_range(sheet,'L3','M22')}
# throw an error if total supply and demand do not match
assert (sum(supply_dict.values()) == sum(demand_dict.values()))

# build the model
# define variables
model = ConcreteModel()
model.num_shipped_per_route = Var(wares_stores, domain = NonNegativeReals)

# define objective function
model.total_cost = Objective( expr = sum(cost_dict[w, s]*model.num_shipped_per_route[w,s]
                                         for (w,s) in wares_stores ),
                            sense = minimize )
model.total_cost.pprint()

# define constraints
model.supply_ct = ConstraintList()

for wh in warehouses:
    model.supply_ct.add(
        sum(model.num_shipped_per_route[w, s] for w,s in wares_stores if w == wh ) == supply_dict[wh])

model.supply_ct.pprint()

model.demand_ct = ConstraintList()
for (sd) in demand_dict.keys():
    model.demand_ct.add(
        sum(model.num_shipped_per_route[w, s] for w, s in wares_stores if s == sd ) == demand_dict.get(sd))

model.demand_ct.pprint()

model.route_cap_ct = ConstraintList()
for w, s in wares_stores:
    model.route_cap_ct.add(model.num_shipped_per_route[w, s] <= cap_dict.get((w, s)))

model.route_cap_ct.pprint()

# solve

solver = SolverFactory('glpk')
solver.solve(model)

# display
print(f"Minimum Total Cost = ${model.total_cost():,.2f}")

print("\nTransported Amounts:")
for (w, s) in wares_stores:
    print(f"Ship {model.num_shipped_per_route[w, s].value:.0f} truckloads from {w} to {s}")

# # or can setup a data frame for nicer display, use zeros for infeasible routes
print("\nData Frame display of transported amounts:")
transp = pd.DataFrame(0, index=warehouses, columns=demand_dict.keys())
for (w, s) in wares_stores:
    transp.loc[w, s] = model.num_shipped_per_route[w, s].value
print(transp)