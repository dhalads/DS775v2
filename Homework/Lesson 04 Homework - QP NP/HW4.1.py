# unfold to see Pyomo solution for Wyndor Quadratic Program
import pyomo.environ as pyo

# Concrete Model
model = pyo.ConcreteModel(name="HW4.1")

predictors = ['x1', 'x2']

bounds_dict = {'x1': (0, None), 'x2': (0, None)}


def bounds_rule(model, pred):
    return (bounds_dict[pred])


model.x = pyo.Var(predictors, domain=pyo.Reals, bounds=bounds_rule)

# Objective
model.limit = pyo.Objective(expr=8.0 * model.x['x1'] -
                         model.x['x1']**2 + 4.0 * model.x['x2'] -
                         model.x['x2']**2.0,
                         sense=pyo.maximize)

model.limit.pprint()

# Constraints
model.Constraint3 = pyo.Constraint(
    expr=model.x['x1'] + model.x['x2'] <= 2)

# Solve
# solver = pyo.SolverFactory('ipopt')
solver = pyo.SolverFactory('/Users/djhalama/anaconda3/envs/ds775/Ipopt-3.13.3-win64-msvs2019-md/bin/ipopt')
solver.solve(model)

# display(model)

# display solution
import babel.numbers as numbers  # needed to display as currency
print(f"\nfunction max = {model.limit():,.2f}")

print(f"x1 = {model.x['x1']():1.2f}")
print(f"x2 = {model.x['x2']():1.2f}")