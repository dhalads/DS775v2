# using openpyxl
from openpyxl import load_workbook
from pyomo.environ import *
import pandas as pd

wb = load_workbook(filename="./data/transp_prob3.xlsx", data_only=True)
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
factories = read_range(sheet, 'A2', 'A6')
warehouses = read_range(sheet, 'B2', 'B11')
stores = read_range(sheet, 'C2', 'C21')
products = read_range(sheet, 'D2', 'D6')
# # stores = ...
# wares_stores = [(w,s) for [w,s] in read_range(sheet,'D3','E31')]
# capacity_dict = ...
route_cost_FW = {(p,f,w):cost for [p,f,w,cost] in read_range(sheet,'G3','J71')}
route_cost_WS = {(p,w,s):cost for [p,w,s,cost] in read_range(sheet,'L3','O202')}
f_w_list = list(set(((f,w)) for [f,w] in read_range(sheet,'H3','I71')))
w_s_list = list(set(((w,s)) for [w,s] in read_range(sheet,'M3','N202')))
p_w_list = list(set(((p,w)) for [p,w] in read_range(sheet,'L3','M202')))
supply_dict = { (p,f):q for [p,f,q] in read_range(sheet,'Q3','S15')}
# demand_dict = ...
demand_dict =  { (p,s):q for [p,s,q] in read_range(sheet,'U3','W102')}
# throw an error if total supply and demand do not match
assert (sum(supply_dict.values()) == sum(demand_dict.values()))

# build the model
# define variables
model = ConcreteModel()
model.num_shipped_per_route_FW = Var(route_cost_FW.keys(), domain = NonNegativeReals)
model.num_shipped_per_route_WS = Var(route_cost_WS.keys(), domain = NonNegativeReals)

# define objective function
model.total_cost = Objective( expr = sum(route_cost_FW[p,f,w]*model.num_shipped_per_route_FW[p,f,w]
                                    for (p,f,w) in route_cost_FW.keys()) +
                                    sum(route_cost_WS[p,w,s]*model.num_shipped_per_route_WS[p,w,s]
                                    for (p,w,s) in route_cost_WS.keys()
                                    ),
                            sense = minimize )
model.total_cost.pprint()

# define constraints
model.supply_ct = ConstraintList()

for p,f in supply_dict.keys():
    model.supply_ct.add(
        sum(model.num_shipped_per_route_FW[p,f,lw] for lp,lf,lw in route_cost_FW if lp == p and lf == f ) == supply_dict[p,f])

model.supply_ct.pprint()

model.demand_ct = ConstraintList()
for p,s in demand_dict.keys():
    model.demand_ct.add(
        sum(model.num_shipped_per_route_WS[p,lw,s] for lp,lw,ls in route_cost_WS if lp == p and ls == s ) == demand_dict[p,s])

model.demand_ct.pprint()

model.CapacityFW_ct = ConstraintList()
for f, w in f_w_list:
    model.CapacityFW_ct.add(
        sum(model.num_shipped_per_route_FW[lp,f, w] for lp,lf,lw in route_cost_FW.keys() if lf == f and lw==w)  <= 1000 )

model.CapacityWS_ct = ConstraintList()
for w, s in w_s_list:
    model.CapacityFW_ct.add(
        sum(model.num_shipped_per_route_WS[lp,w,s] for lp,lw,ls in route_cost_WS.keys() if lw == w and ls==s)  <= 800 )

model.MaxStorage_ct = ConstraintList()
for w in warehouses:
    model.MaxStorage_ct.add(
        sum(model.num_shipped_per_route_FW[lp, lf, w] for lp,lf,lw in route_cost_FW.keys() if lw==w)  <= 2300 )

model.warehouse_in_out_ct = ConstraintList()
for p, w in p_w_list:
    model.MaxStorage_ct.add(
        sum(model.num_shipped_per_route_FW[p, lf, w] for lp,lf,lw in route_cost_FW.keys() if lp==p and lw==w)  ==
        sum(model.num_shipped_per_route_WS[p, w, ls] for lp,lw,ls in route_cost_WS.keys() if lp==p and lw==w) )
# # solve

solver = SolverFactory('glpk')
solver.solve(model)

# # display
print(f"Minimum Total Cost = ${model.total_cost():,.2f}")

# # or can setup a data frame for nicer display, use zeros for infeasible routes
print("\nData Frame display of transported amounts:")
df = pd.DataFrame(columns = ['Product' , 'Start', 'End' , 'Units'])
for p,f,w in route_cost_FW:
    df = df.append({'Product' : p , 'Start' : f, 'End' : w, "Units" : model.num_shipped_per_route_FW[p, f, w].value } , ignore_index=True)
    # df = df.append( pd.Series((p , f, w, model.num_shipped_per_route_FW[p, f, w].value )) , ignore_index=True)
for p,w,s in route_cost_WS:
    df = df.append({'Product' : p , 'Start' : w, 'End' : s, "Units" : model.num_shipped_per_route_WS[p, w, s].value } , ignore_index=True)
print(df)