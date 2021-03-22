# EXECUTE FIRST

# computational imports
from pyomo.environ import *
# for reading files from urls
import urllib.request
# display imports
from IPython.display import display, IFrame
from IPython.core.display import HTML

# abstract Good Products

# Problem data
products = ['door', 'window']
unit_profit = dict(zip(products, [3,5]))

sales_potential = dict(zip(products, [7, 5, 9]))
def bounds_rule(model, product):
    return ((0, sales_potential[product]))

plants = ['Plant1', 'Plant2', 'Plant3']
production_avail = dict(zip(plants, [4, 12, 18]))

tpu = [[1,0], [0,2], [3,2]]
time_per_unit = {
    plants[p]: dict(zip(products, tpu[p][:]))
    for p in range(len(plants))
}
bigM = 10000

max_num_products_to_make = 1

# Instantiate concrete model
M = ConcreteModel(name="Example_1")

# Decision Variables
# M.x = Var(products, domain=Reals, bounds=bounds_rule)
M.x = Var( products, domain = NonNegativeReals)
M.y = Var(products, domain=Boolean)
# M.plant_choice = Var(plants, domain=Boolean)

# Objective:  Maximize Profit
M.profit = Objective(expr=sum(unit_profit[pr] * M.x[pr] for pr in products),
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
            for pr in products) <= production_avail[pl])

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