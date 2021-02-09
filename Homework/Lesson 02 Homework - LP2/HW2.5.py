from pyomo.environ import *

# abstract Wyndor
dailyReq = ['Carbs', 'Protein', 'Vitamins']
feeds = ['Corn', 'Tankage', 'Alfalfa']
cost_rate = {'Corn': 2.1, 'Tankage': 1.8, 'Alfalfa':1.5}
min_nutrients = {'Carbs': 200, 'Protein': 180, 'Vitamins': 150}
nutrients_per_kg = {
    'Corn': {
        'Carbs': 90,
        'Protein': 30,
        'Vitamins': 10
    },
    'Tankage': {
        'Carbs': 20,
        'Protein': 80,
        'Vitamins': 20
    },
    'Alfalfa': {
        'Carbs': 40,
        'Protein': 60,
        'Vitamins': 60
    }
}

#Concrete Model
model = ConcreteModel()

#Decision Variables
model.weekly_feed = Var(feeds, domain=NonNegativeReals)

#Objective
model.cost = Objective(expr=sum(cost_rate[fd] * model.weekly_feed[fd]
                               for fd in feeds),
                      sense=minimize)

model.capacity = ConstraintList()
for dr in dailyReq:
    model.capacity.add(
        sum(nutrients_per_kg[fd][dr] * model.weekly_feed[fd]
            for fd in feeds) >= min_nutrients[dr])

# Solve
solver = SolverFactory('glpk')
solver.solve(model)

# display solution (again, we've changed to f-strings)
print(f"Min cost = ${model.cost():,.2f}")
for j in feeds:
    print(f"Kg's of {j} = {model.weekly_feed[j]():.1f}")