# now the solution to 3.4-10 for decison variables restricted to integers

from pyomo.environ import *
# from glpk import *
import sys
print (sys.prefix)

# Concret Model
model = ConcreteModel(name = "Staffing")

# Decision Variables
model.x = Var( ['x1m','x1t','x1w','x1r','x1f',
                'x2m','x2t','x2w','x2r','x2f',
                'x3m','x3t','x3w','x3r','x3f',
                'x4m','x4t','x4w','x4r','x4f',
                'x5m','x5t','x5w','x5r','x5f',
                'x6m','x6t','x6w','x6r','x6f'], 
              domain = NonNegativeReals)

print(model.x)

# Objective
model.obj = Objective( expr = 25 * (model.x['x1m'] + model.x['x1t'] + model.x['x1w']+ model.x['x1r']+ model.x['x1f']) +
                      26 * (model.x['x2m'] + model.x['x2t'] + model.x['x2w']+ model.x['x2r']+ model.x['x2f']) +
                      24 * (model.x['x3m'] + model.x['x3t'] + model.x['x3w']+ model.x['x3r']+ model.x['x3f']) +
                      23  * (model.x['x4m'] + model.x['x4t'] + model.x['x4w']+ model.x['x4r']+ model.x['x4f']) +
                      28  * (model.x['x5m'] + model.x['x5t'] + model.x['x5w']+ model.x['x5r']+ model.x['x5f']) +
                      30  * (model.x['x6m'] + model.x['x6t'] + model.x['x6w']+ model.x['x6r']+ model.x['x6f'])
                , sense = minimize)

# # Constraints
model.MinHour1 = Constraint( expr = model.x['x1m'] + model.x['x1t'] + model.x['x1w']+ model.x['x1r']+ model.x['x1f'] >= 7 )
model.MinHour2 = Constraint( expr = model.x['x2m'] + model.x['x2t'] + model.x['x2w']+ model.x['x2r']+ model.x['x2f'] >= 7 )
model.MinHour3 = Constraint( expr = model.x['x3m'] + model.x['x3t'] + model.x['x3w']+ model.x['x3r']+ model.x['x3f'] >= 7 )
model.MinHour4 = Constraint( expr = model.x['x4m'] + model.x['x4t'] + model.x['x4w']+ model.x['x4r']+ model.x['x4f'] >= 7 )
model.MinHour5 = Constraint( expr = model.x['x5m'] + model.x['x5t'] + model.x['x5w']+ model.x['x5r']+ model.x['x5f'] >= 8 )
model.MinHour6 = Constraint( expr = model.x['x6m'] + model.x['x6t'] + model.x['x6w']+ model.x['x6r']+ model.x['x6f'] >= 8 )

model.MaxHour_m = Constraint( expr = model.x['x1m'] + model.x['x2m'] + model.x['x3m']+ model.x['x4m']+ model.x['x5m']+ model.x['x6m'] == 14 )
model.MaxHour_t = Constraint( expr = model.x['x1t'] + model.x['x2t'] + model.x['x3t']+ model.x['x4t']+ model.x['x5t']+ model.x['x6t'] == 14 )
model.MaxHour_w = Constraint( expr = model.x['x1w'] + model.x['x2w'] + model.x['x3w']+ model.x['x4w']+ model.x['x5w']+ model.x['x6w'] == 14 )
model.MaxHour_r = Constraint( expr = model.x['x1r'] + model.x['x2r'] + model.x['x3r']+ model.x['x4r']+ model.x['x5r']+ model.x['x6r'] == 14 )
model.MaxHour_f = Constraint( expr = model.x['x1f'] + model.x['x2f'] + model.x['x3f']+ model.x['x4f']+ model.x['x5f']+ model.x['x6f'] == 14 )

model.Max1m = Constraint( expr = model.x['x1m'] <= 6)
model.Max1t = Constraint( expr = model.x['x1t'] <= 0)
model.Max1w = Constraint( expr = model.x['x1w'] <= 6)
model.Max1r = Constraint( expr = model.x['x1r'] <= 0)
model.Max1f = Constraint( expr = model.x['x1f'] <= 6)

model.Max2m = Constraint( expr = model.x['x2m'] <= 0)
model.Max2t = Constraint( expr = model.x['x2t'] <= 6)
model.Max2w = Constraint( expr = model.x['x2w'] <= 0)
model.Max2r = Constraint( expr = model.x['x2r'] <= 6)
model.Max2f = Constraint( expr = model.x['x2f'] <= 0)

model.Max3m = Constraint( expr = model.x['x3m'] <= 4)
model.Max3t = Constraint( expr = model.x['x3t'] <= 8)
model.Max3w = Constraint( expr = model.x['x3w'] <= 4)
model.Max3r = Constraint( expr = model.x['x3r'] <= 0)
model.Max3f = Constraint( expr = model.x['x3f'] <= 4)

model.Max4m = Constraint( expr = model.x['x4m'] <= 5)
model.Max4t = Constraint( expr = model.x['x4t'] <= 5)
model.Max4w = Constraint( expr = model.x['x4w'] <= 5)
model.Max4r = Constraint( expr = model.x['x4r'] <= 0)
model.Max4f = Constraint( expr = model.x['x4f'] <= 5)


model.Max5m = Constraint( expr = model.x['x5m'] <= 3)
model.Max5t = Constraint( expr = model.x['x5t'] <= 0)
model.Max5w = Constraint( expr = model.x['x5w'] <= 3)
model.Max5r = Constraint( expr = model.x['x5r'] <= 8)
model.Max5f = Constraint( expr = model.x['x5f'] <= 0)

model.Max6m = Constraint( expr = model.x['x6m'] <= 0)
model.Max6t = Constraint( expr = model.x['x6t'] <= 0)
model.Max6w = Constraint( expr = model.x['x6w'] <= 0)
model.Max6r = Constraint( expr = model.x['x6r'] <= 6)
model.Max6f = Constraint( expr = model.x['x6f'] <= 2)



# # Solve
solver = SolverFactory('glpk')
solver.solve(model)

# remove the comment symbol to see the pyomo display of results
# display(model)

print(model.obj()) # added by Dr. B.

print(f"Person: Hours")
print(f"Hours in format (m, t, w, r, f)")
print(f"K.C.:({model.x['x1m']()}, {model.x['x1t']()}, {model.x['x1w']()}, {model.x['x1r']()}, {model.x['x1f']()} )")
print(f"D.H.:({model.x['x2m']()}, {model.x['x2t']()}, {model.x['x2w']()}, {model.x['x2r']()}, {model.x['x2f']()} )")
print(f"H.B.:({model.x['x3m']()}, {model.x['x3t']()}, {model.x['x3w']()}, {model.x['x3r']()}, {model.x['x3f']()} )")
print(f"S.C.:({model.x['x4m']()}, {model.x['x4t']()}, {model.x['x4w']()}, {model.x['x4r']()}, {model.x['x4f']()} )")
print(f"K.S.:({model.x['x5m']()}, {model.x['x5t']()}, {model.x['x5w']()}, {model.x['x5r']()}, {model.x['x5f']()} )")
print(f"N.K.:({model.x['x6m']()}, {model.x['x6t']()}, {model.x['x6w']()}, {model.x['x6r']()}, {model.x['x6f']()} )")
