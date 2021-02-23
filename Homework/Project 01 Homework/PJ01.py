# EXECUTE FIRST

# computational imports
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pyomo.environ import *


airfares = pd.read_csv("./data/Airfares.csv")

inputs = ['COUPON', 'HI', 'DISTANCE']
lm_outputs = ['PAX', 'S_INCOME', 'E_INCOME', 'FARE']
lm_dict ={}

def createLinearModels(inputs, lm_outputs):
    for out in lm_outputs:
        X = airfares[inputs]
        Y = airfares[out]
        model = sm.OLS(Y, X).fit()
        lm_dict.update({out: model})

def printlm():
    for lm in lm_dict.keys():
        print(f"\nModel for {lm}")
        print(lm_dict.get(lm).summary())

createLinearModels(inputs, lm_outputs)
printlm()
model = ConcreteModel()

model.inputs = Var(inputs,  domain=NonNegativeReals)

# model.max_fare = Objective(expr=lm_dict['FARE'].predict([model.inputs['COUPON'], model.inputs['HI'], model.inputs['DISTANCE']])[0],
#                              sense=maximize)
model.max_fare = Objective(expr=lm_dict['FARE'].predict([i for i in model.inputs.values()])[0],
                            sense=maximize)
model.max_fare.pprint()

model.pax_con = Constraint( expr=lm_dict['PAX'].predict([model.inputs['COUPON'], model.inputs['HI'], model.inputs['DISTANCE']])[0] <= 20000 )
model.pax_con.pprint()

model.s_income_con = Constraint( expr=lm_dict['S_INCOME'].predict([model.inputs['COUPON'], model.inputs['HI'], model.inputs['DISTANCE']])[0] <= 30000 )
model.s_income_con.pprint()

model.e_income_con = Constraint( expr=lm_dict['E_INCOME'].predict([model.inputs['COUPON'], model.inputs['HI'], model.inputs['DISTANCE']])[0] >= 30000 )
model.e_income_con.pprint()

model.coupon_con = Constraint( expr=(0, model.inputs['COUPON'], 1.5) )
model.coupon_con.pprint()

model.hi_con = Constraint( expr=(4000, model.inputs['HI'], 8000) )
model.hi_con.pprint()

model.distance_con = Constraint( expr=(500, model.inputs['DISTANCE'], 1000) )
model.distance_con.pprint()

solver = SolverFactory('glpk')
solver.solve(model)

# display
print(f"Max fare = ${model.max_fare():,.2f}")

#inputs
for x in model.inputs.keys():
    print(f"{x}={model.inputs[x].value}")

model.write('model.lp', io_options={'symbolic_solver_labels': True})
#!glpsol -m model.lp --lp --ranges sensit.sen
command = "glpsol -m model.lp --lp --ranges sensit.sen > glpsol.out"
import subprocess
subprocess.call(command)

import numpy as np
# print sensitivity report
np.set_printoptions(linewidth=110)
f = open('sensit.sen', 'r')
file_contents = f.read()
print(file_contents)
f.close()