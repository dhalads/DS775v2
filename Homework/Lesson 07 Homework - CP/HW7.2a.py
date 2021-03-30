# computational imports
from pyomo.environ import *
# for reading files from urls
import urllib.request
# display imports
from IPython.display import display, IFrame
from IPython.core.display import HTML

dist = ['1', '2', '3']
units_dist = dict(zip(dist, [20,20,20]))
# product_startup_cost = dict(zip(products, [50000,40000,70000,60000]))
plants = ['A','B']
cpu = [ [600, 700, 400], [700, 800, 500] ]
cost_per_unit = {
    plants[p]: dict(zip(dist, cpu[p][:]))
    for p in range(len(plants))
}

production_avail = dict(zip(plants, [ 50, 50]))

bigM = 100000

max_num_products_to_make = 2
max_num_plants = 1

# Instantiate concrete model
M = ConcreteModel(name="HW7.2a")

# Decision Variables
# M.x = Var(products, domain=Reals, bounds=bounds_rule)
M.x = Var(plants, domain = NonNegativeReals)
M.y = Var(plants,dist, domain=NonNegativeReals)
M.x.pprint()
M.y.pprint()



# Objective:  Maximize Profit
M.cost = Objective(expr=sum(cost_per_unit[p][d]*M.y[p, d] for d in dist for p in plants),
                     sense=minimize)
M.cost.pprint()

# Constraints:
M.constraints = ConstraintList()

for p in plants:  # produce product only if product is chosen
    M.constraints.add(M.x[p] <= production_avail[p])

for p in plants:  # trucks shipped from each plant
    M.constraints.add(sum(M.y[p,d] for d in dist) == M.x[p])

for d in dist:  # trucks shipped to each dist
    M.constraints.add(sum(M.y[p,d] for p in plants) == units_dist[d])

M.constraints.add(sum(M.x[p] for p in plants) <= 60)

M.constraints.pprint()

# Solve
solver = SolverFactory('glpk')
solver.solve(M)

print(f"\nMinimum cost = ${M.cost():,.2f}")

print("\nWhich plant to use:")
for p in plants:
    print(f"At plant {p} produce amount {M.x[p].value}")

print("\nWhat to ship from plant to disttribution center")
for p in plants:
    for d in dist:
        print(f"Ship from {p} to {d} : {M.y[p,d].value}")
