# EXECUTE FIRST

# computational imports
from pyomo.environ import *
import numpy as np
import pandas as pd
# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
# for reading files from urls
import urllib.request
import statistics as st
# display imports
from IPython.display import display, IFrame
from IPython.core.display import HTML

# for playing videos, customize height and width if desired
def play_video(vid_name, w = 640, h = 360):
    vid_path = "https://media.uwex.edu/content/ds/ds775_r19/"
    return IFrame( vid_path + vid_name + "/index.html", width = w, height = h )

# import notebook styling for tables and width etc.
response = urllib.request.urlopen('https://raw.githubusercontent.com/DataScienceUWL/DS775v2/master/ds755.css')
HTML(response.read().decode("utf-8"));

# abstract Sausage Factory

### Problem Data ###
def runOpt(demandecon, demandprem, discountpork):
    types = ['economy','premium']
    ingredients = ['pork', 'wheat', 'starch']

    cost = dict( zip( ingredients, [4.32, 2.46, 1.86] ) )

    kg_per_sausage = 0.05
    number_each_type = dict( zip( types, [demandecon, demandprem] ) )

    mnpi = [[.4,.6],[0,0],[0,0]]
    min_prop_ing = { ingredients[i]:{ types[j]:mnpi[i][j] for j in range(len(types)) } for i in range(len(ingredients)) }
    mxpi = [[1,1],[1,1],[.25,.25]]
    max_prop_ing = { ingredients[i]:{ types[j]:mxpi[i][j] for j in range(len(types)) } for i in range(len(ingredients)) }

    max_ingredient = dict( zip( ingredients, [100, 20, 17] ) )
    min_ingredient = dict( zip( ingredients, [discountpork,  0,  0] ) )

    ### Pyomo Model ###

    # Concrete Model
    M = ConcreteModel(name = "Sausages")

    # Decision Variables
    M.amount = Var(ingredients, types, domain = NonNegativeReals)

    # Objective
    M.cost = Objective( expr = sum( cost[i] * sum(M.amount[i,t] for t in types)
                                for i in ingredients)  - 1.22*discountpork, sense = minimize )

    M.tot_sausages_ct = ConstraintList()
    for t in types:
        M.tot_sausages_ct.add( sum( M.amount[i,t] for i in ingredients )
                            == kg_per_sausage * number_each_type[t] )

    M.min_prop_ct = ConstraintList()
    for i in ingredients:
        for t in types:
            M.min_prop_ct.add( M.amount[i,t] >= min_prop_ing[i][t] *
                            sum( M.amount[k,t] for k in ingredients ) )

    M.max_prop_ct = ConstraintList()
    for i in ingredients:
        for t in types:
            M.max_prop_ct.add( M.amount[i,t] <= max_prop_ing[i][t] * 
                            sum( M.amount[k, t] for k in ingredients ) )

    M.max_ingredient_ct = ConstraintList()
    for i in ingredients:
        M.max_ingredient_ct.add( sum( M.amount[ i, t] for t in types ) <= 
                            max_ingredient[i] )
        
    M.min_ingredient_ct = ConstraintList()
    for i in ingredients:
        M.min_ingredient_ct.add( sum( M.amount[ i, t] for t in types ) >=
                            min_ingredient[i] )

    ### Solution ###
    solver = SolverFactory('glpk')
    solver.solve(M)

    ### Display ###
    # print(f"Total Cost = ${M.cost():,.2f}")

    # put amounts in dataframe for nicer display
    import pandas as pd
    dvars = pd.DataFrame( [ [M.amount[i,t]() for t in types] for i in ingredients ],
                        index = ['Pork','Wheat','Starch'],
                        columns = ['Economy','Premium'])
    # print("Kilograms of each ingredient in each type of sausage:")
    # print(dvars)
    totalPork = M.amount['pork','economy']() + M.amount['pork','premium']()
    amountFullPricePork = totalPork - discountpork
    output = (amountFullPricePork, M.cost())
    return output

# runOpt(361, 544, 20)

def problemA(discountPork):
    results = []
    for i in range(10):
        econDemand = np.random.randint(325, 375,1)
        premDemand = np.random.randint(450, 550,1)
        # print(f"{(econDemand[0], premDemand[0], discountPork)}")
        result = runOpt(econDemand[0], premDemand[0], discountPork)
        results.append(result)
    return (results)

# prob1 = problemA(20)
# print(prob1)

# fullPork = [i[0] for i in prob1]
# cost = [i[1] for i in prob1]
# plt.figure(figsize = (8,5))
# # display Winnings in a histogram
# plt.hist(fullPork,color="olive")
# plt.ylabel('Frequency')
# plt.xlabel('Kg Full Price Pork')
# plt.show()

# plt.hist(cost,color="olive")
# plt.ylabel('Frequency')
# plt.xlabel('Cost')
# plt.show()

def ProblemC():
    output = []
    preOrder = [(17 + i) for i in range(11)]
    for j in preOrder:
        result = problemA(j)
        fullPork = [i[0] for i in result]
        cost = [i[1] for i in result]
        tmp = (j, st.mean(fullPork), np.percentile(fullPork, 5), np.percentile(fullPork, 95), st.mean(cost), np.percentile(cost, 5), np.percentile(cost, 95))
        output.append(tmp)
    return(output)

result = ProblemC()
print(result)
import pandas as pd
dvars = pd.DataFrame( result,
                    columns = ['Pre-Order','MeanPork','5Pork','95Pork','MeanCost','5Cost','95Cost'])
print(dvars)

# plot results and trend chart showing middle 90% of simulated profits for each order quantity
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot('Pre-Order','MeanCost',data=dvars, linestyle='-', marker='o')
plt.xlabel('Discount Pork Ordered', fontsize=12)
plt.ylabel('Mean Cost', fontsize=12)
plt.title("Trend Chart")
ax.fill_between('Pre-Order','5Cost','95Cost',data=dvars,color="#b9cfe7", edgecolor="#b9cfe7")
plt.show()