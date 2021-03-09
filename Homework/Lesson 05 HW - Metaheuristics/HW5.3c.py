import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from scipy.optimize import minimize
import json
from simanneal import Annealer

pop_size = 100 # should be even due to the way we'll implement crossover
ind_size = 10 # determines number of input variables for Rastrigin and each individual
lower = -5.12 # lower and upper bounds on the real variables
upper = 5.12
bounds = [(lower,upper) for i in range(ind_size)]
tourn_size = 3 # tournament size for selection
cx_prob = 0.8 # probability a pair of parents crossover to produce two children
mut_prob = 0.2 # probability an individual mutates
ind_prob = 0.1 # probability each variable in an individual mutates
sigma = (upper-lower)/6 # standard deviation (scale) for gaussian mutations
num_iter = 2000 # number of genetic algorithm mutations
update_iter = 100 # how often to display output
alpha = 0.2

stats = np.zeros((num_iter+1,3)) # for collecting statistics

# Create function for blended crossover
def create_blended_child(x, y):
    length = len(x)
    z = np.zeros(length)
    for i in range(length):
        z_min = min(x[i], y[i])
        z_max = max(x[i], y[i])
        z_range = abs(x[i]-y[i])
        zlow = z_min - alpha * z_range
        zhigh = z_max + alpha * z_range
        z[i] = np.random.uniform(zlow, zhigh)
    return(z)

# objective or fitness function
def rastrigin(x):
    x = np.array(x) # force a numpy arrray here so that the math below works
    return np.sum(x**2 + 10 - 10 * np.cos(2 * np.pi * x) )

#initialize population and fitness
pop = np.random.uniform(low=lower, high=upper, size = (ind_size,pop_size))
fitness = np.zeros(pop_size)
for j in range(pop_size):
    fitness[j] = rastrigin(pop[:,j])

# initialize stats and output
best_fitness = min(fitness)
stats[0,:] = np.array([0,best_fitness, best_fitness])
print('Iteration | Best this iter |    Best ever')

for iter in range(num_iter):
    ### CHANGE - local search goes here
    #    - sort pop by increasing fitness
    #    - take first three individuals with lowest fitness and replace them by the minimizing location resulting 
    #      from using scipy.optimize.minimize with bounds applied to each individual each individual




    # tournament selection
    sorted_pos = fitness.argsort() # sort pop by increasing fitness
    fitness = fitness[sorted_pos]
    pop = pop[:,sorted_pos]
    for i in range(3):
        result = minimize(rastrigin, pop[:,i], bounds=bounds)
        pop[:,i] = result.x
        fitness[i] = result.fun
    select_pop = np.zeros((ind_size,pop_size)) # initialize selected population
    for j in range(pop_size):
        subset_pos = np.random.choice(pop_size,tourn_size,replace=False) # select without replacement
        smallest_pos = np.min(subset_pos) # choose index corresponding to lowest fitness
        select_pop[:,j] = pop[:,smallest_pos]

    ### CHANGE this to blended crossover
    # one-point crossover (mating)
    cx_pop = np.zeros((ind_size,pop_size)) # initialize crossover population
    for j in range(int(pop_size/2)):  # pop_size must be even
        parent1, parent2 = select_pop[:,2*j], select_pop[:,2*j+1]
        child1, child2 = parent1.copy(), parent2.copy()
        if np.random.uniform() < cx_prob:
            child1 = create_blended_child(parent1, parent2)
            child2 = create_blended_child(parent1, parent2)
        cx_pop[:,2*j] = child1
        cx_pop[:,2*j+1] = child2

    # gaussian mutation (rewritten to remove nested loop for speed)
    mut_pop = np.zeros((ind_size,pop_size)) # initialize mutation population
    for j in range(pop_size):
        individual = cx_pop[:,j].copy() # copy is necessary to avoid conflicts in memory
        if np.random.uniform()<mut_prob:
            individual = individual + np.random.normal(0,sigma,ind_size)*(np.random.uniform(size=ind_size)<ind_prob)
            individual = np.maximum(individual,lower) # clip to lower bound
            individual = np.minimum(individual,upper) # clip to upper bound
        mut_pop[:,j] = individual.copy() # copy is necessary to avoid conflicts in memory

    # fitness evaluation with local search
    pop = mut_pop.copy()
    for j in range(pop_size):
        fitness[j] = rastrigin(pop[:,j])

    # collect stats and output to screen
    min_fitness = min(fitness) # best for this iteration
    if min_fitness < best_fitness: # best for all iterations
        best_fitness = min_fitness
        index = np.argmin(fitness)
        best_x = pop[:,index]

    stats[iter+1,:] = np.array([iter+1,min_fitness, best_fitness])
    if (iter+1) % update_iter == 0:
        print(f"{stats[iter+1,0]:9.0f} | {stats[iter+1,1]:14.3e} | {stats[iter+1,2]:12.3e}")
        
print(f"The minimum value found of the Rastrigin function is {best_fitness:.4f}")
print("The location of that minimum is:")
print('(',', '.join(f"{x:.4f}" for x in best_x),')')