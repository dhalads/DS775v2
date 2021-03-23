# computational imports
from pyomo.environ import *
# for reading files from urls
import urllib.request
# display imports
from IPython.display import display, IFrame
from IPython.core.display import HTML

routes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
route_times = dict(zip(routes, [6, 4, 7, 5, 4, 6, 5, 3, 7, 6]))

locations = ['A','B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
dpl = [ [1,0,0,1,0,0,1,0,0], [0,1,0,0,0,1,0,0,1], [0,0,1,0,1,0,0,1,0],
    [0,1,1,0,1,0,0,0,1], [1,0,0,0,0,1,0,1,0], [0,1,0,1,1,0,0,0,0],
    [0,0,1,0,0,0,1,0,1], [0,0,0,1,0,0,1,0,0], [1,1,1,0,0,0,0,0,0], [0,1,0,0,0,0,1,1,0]]
locs_per_route = {
    routes[r]: dict(zip(locations, dpl[r][:]))
    for r in range(len(routes))
}

# Instantiate concrete model
M = ConcreteModel(name="Example_1")

# Decision Variables
M.r = Var(routes, domain=Boolean)


# Objective:  Maximize Profit
M.time = Objective(expr=sum(route_times[r]*M.r[r] for r in routes),
                     sense=minimize)
M.time.pprint()

# Constraints:
M.constraints = ConstraintList()

# choose only 3 routes
M.constraints.add(sum(M.r[r] for r in routes) == 3)

#only 1 route per location
for l in locations:
    M.constraints.add(
        sum(locs_per_route[r][l] * M.r[r]
            for r in routes) <= 1)


M.constraints.pprint()

# Solve
solver = SolverFactory('glpk')
solver.solve(M)

print(f"\nMinimum time(minutes) is {M.time()}")

print("\nWhich routes to use:")
for r in routes:
    print(f"Use route {r}? {['No','Yes'][int(M.r[r]())]}")

