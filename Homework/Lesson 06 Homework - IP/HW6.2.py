# computational imports
from pyomo.environ import *
# for reading files from urls
import urllib.request
# display imports
from IPython.display import display, IFrame
from IPython.core.display import HTML

products = ['1', '2', '3', '4']
unit_profit = dict(zip(products, [70,60,90,80]))
product_startup_cost = dict(zip(products, [50000,40000,70000,60000]))
plants = ['1','2']
tpu = [ [5, 3, 6, 4], [4, 6, 3, 5] ]
time_per_unit = {
    plants[p]: dict(zip(, tpu[p][:]))
    for p in range(len(plants))
}
production_avail = dict(zip(plants, [ 6000, 6000]))

binCon = ['1', '2', '3']

ycoef = [ [1, 1, 1, 1], [-1,-1,1,0] , [-1,-1,0,1]]
coef_per_cst = {
    binCon[bc]: dict(zip(, ycoef[bc][:]))
    for bc in range(len(binCon))
}
ycoeflimit = dict(zip(intCon, [2, 0, 0]))

bigM = 100000

max_num_products_to_make = 1

# Instantiate concrete model
M = ConcreteModel(name="Example_1")

# Decision Variables
# M.x = Var(products, domain=Reals, bounds=bounds_rule)
M.x = Var( products, domain = NonNegativeReals)
M.y = Var(products, domain=Boolean)
M.plant_choice = Var(plants, domain=Boolean)

# Objective:  Maximize Profit
M.profit = Objective(expr=sum(unit_profit[pr] * M.x[pr] - product_startup_cost[pr]*M.y[pr]for pr in products),
                     sense=maximize)
M.profit.pprint()

# Constraints:
M.constraints = ConstraintList()

for pr in products:  # produce product only if product is chosen
    M.constraints.add(M.x[pr] <= bigM * M.y[pr])

# choose 2 products
M.constraints.add(sum(M.y[pr] for pr in products) <= max_num_products_to_make)

for pl in plants:  # production capacities
    M.constraints.add(
        sum(time_per_unit[pl][pr] * M.x[pr]
            for pr in products) <= production_avail[pl] + bigM*M.plant_choice[pl])

for bc in binCon:  # production capacities
    M.constraints.add(
        sum(time_per_unit[bc][ycoef] * M.y[pr]
            for pr in products) <= production_avail[pl] + bigM*M.plant_choice[pl])

M.constraints.pprint()

# Solve
solver = SolverFactory('glpk')
solver.solve(M)

print(f"\nMaximum Profit = ${1000 * M.profit():,.2f}")

# print("\nWhich plant to use:")
# for pl in plants:
#     print(f"Produce at {pl}? {['No','Yes'][int(M.plant_choice[pl]())]}")

print("\nWhich products and how many:")
for pr in products:
    if bool(M.y[pr]()):
        print(f"Produce {pr} at a rate of {M.x[pr]():.2f} batches per week")
    else:
        print(f"Do not produce {pr}" )