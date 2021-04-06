# EXECUTE FIRST

# computational imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
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

# helper function for nicely printing dollar amounts
def dollar_print(x):
    if x < 0:
        return(f"-${-x:,.2f}")
    else:
        return(f"${x:,.2f}")


results = []

for j in range(1000):
    stock = []
    bond = []
    stock.append(5000)
    bond.append(5000)
    for i in range(6):
        intbond = np.random.normal(loc=4,scale=3,size=1)
        intstock = np.random.normal(loc=8,scale=6,size=1)
        sumbond = sum(bond)
        sumstock = sum(stock)
        bond.append((sumbond*intbond/100)[0])
        stock.append((sumstock*intstock/100)[0])
        bond.append(2000)
        stock.append(2000)
    # print(f"stock = {stock}")
    # print(f"bond = {bond}")
    results.append((sum(bond)+sum(stock)))

# print(f"Number of tries {len(results)} is  {results}")
print(f"Number of tries {len(results)}")
print(f"The mean of college fund {dollar_print(st.mean(results))}")
print(f"The standard deviation of college fund {dollar_print(st.stdev(results))}")
num35k = sum([1 for i in results if i>=35000])
num40k = sum([1 for i in results if i>=40000])
print(f"The probability >= $35,000  {num35k/len(results)}")
print(f"The probability >= $40,000  {num40k/len(results)}")

