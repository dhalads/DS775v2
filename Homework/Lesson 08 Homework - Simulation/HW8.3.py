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

# import matplotlib.pyplot as plt
# h = plt.hist(np.random.triangular(-3, 0, 8, 100000), bins=200,
#              density=True)
# plt.show()


rpicost = 5.05

def bidAnalysis(rpibid):
    bids = []
    for j in range(1000):
        compbids = np.random.triangular(5.25, 6, 7, 4)
        bids.append(max(compbids))
    numWinBids = [i for i in bids if i<=rpibid]
    numWin = len(numWinBids)
    probWin = numWin/len(bids)
    if numWin ==0 :
        meanProfit = 0.0
    else:
        meanProfit = sum([(i-rpicost) for i in numWinBids])/len(numWinBids)
    return(numWin, probWin, meanProfit)

def runProblem1():
    numWin, probWin, meanProfit = bidAnalysis(5.7)
    print(f"Number of bids to win is {numWin}")
    print(f"Probability to win bid is {probWin}")
    print(f"Mean profit in millions is {dollar_print(meanProfit)}")

# runProblem1()

def runProblem2():
    newbids = [(5.3 + 0.1*i) for i in range(8)]
    meanProfits = [bidAnalysis(i)[2] for i in newbids]
    maxProfit = max(meanProfits)
    index = meanProfits.index(maxProfit)
    bestBid = newbids[index]
    print(f"Best bid is {bestBid} for profit of {dollar_print(maxProfit)} million")

# runProblem2()

def runProblem3():
    newbids = [(5.3 + 0.1*i) for i in range(8)]
    meanProfits = [bidAnalysis(i)[2] for i in newbids]
    plt.scatter(newbids, meanProfits)
    plt.title('Trend of profit vs bid')
    plt.xlabel('bid(millions)')
    plt.ylabel('profit(millions)')
    plt.show()

runProblem3()

# plt.figure(figsize = (8,5))
# # display Winnings in a histogram
# plt.hist(a,color="olive")
# plt.ylabel('Frequency')
# plt.xlabel('Normal(20,4) Variable')

# print(f"Number of tries {len(bids)} is  {bids}")
# print(f"Number of tries {len(bids)}")

# print(f"Number of bids to win is {len(numWin)}")
# print(f"Probability to win bid is {len(numWin)/len(bids)}")

# print(f"Mean profit in millions is {dollar_print(meanProfit)}")
# print(f"The mean of college fund {dollar_print(st.mean(results))}")
# print(f"The standard deviation of college fund {dollar_print(st.stdev(results))}")
# num35k = sum([1 for i in results if i>=35000])
# num40k = sum([1 for i in results if i>=40000])
# print(f"The probability >= $35,000  {num35k/len(results)}")
# print(f"The probability >= $40,000  {num40k/len(results)}")
