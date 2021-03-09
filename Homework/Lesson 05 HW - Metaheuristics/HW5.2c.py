import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from scipy.optimize import minimize
import json
from simanneal import Annealer

#####
pop_size = 10 # should be even due to the way we'll implement crossover
ind_size = 20 # determines number of input variables for each tour
######
#lower = -5.12 # lower and upper bounds on the real variables
#upper = 5.12
tourn_size = 3 # tournament size for selection
cx_prob = 0.8 # probability a pair of parents crossover to produce two children
mut_prob = 0.3 # probability an individual mutates
ind_prob = 0.1 # probability each variable in an individual mutates
#sigma = (upper-lower)/6 # standard deviation (scale) for gaussian mutations
###### maybe here
num_iter = 1000 # number of genetic algorithm mutations
update_iter = 100 # how often to display output
######
np.random.seed(123)
stats = np.zeros((num_iter+1,3)) # for collecting statistics

def create_set1():
    num_items = 20
    np.random.seed(seed=123)
    values = np.random.randint(low=5, high=50, size=num_items)
    weights = np.random.randint(low=1, high=10, size=num_items)
    x = np.zeros(num_items, dtype = bool)  # all false
    return values, weights, x

def print_set(label):
    weight = sum( weights[x] )
    value = sum( values[x] )
    print(label)
    print(f"items selected: {x}")
    print(f"values: {values}")
    print(f"weights: {weights}")



def print_out(label):
    weight = sum( weights[x] )
    value = sum( values[x] )
    print(label)
    print(f"items selected: {x}")
    print(f"value: {value}")
    print(f"weight: {weight}")

max_weight = 50
values, weights, x = create_set1()
np.random.seed() # use system clock to reset the seed so future random numbers will appear random


######
# objective or fitness function
def knapsack_value_penalty(x, values, weights, max_tot_weight):
    # x is a vector of booleans of which items to include
    tot_value = sum(values[x])
    penalty = sum(values)*min( max_tot_weight - sum(weights[x]), 0) 
    return -(tot_value+penalty)
######

###### may need to also enforce that the variables are integers and not floats
#initialize population and fitness np.random.randint(0,2,size=20,dtype=bool)
# init_tour = np.random.permutation(np.arange(len(distance_matrix))).astype(int).tolist()
# pop = np.random.uniform(low=lower, high=upper, size = (ind_size,pop_size))
pop = np.zeros((ind_size, pop_size)).astype(int)
fitness = np.zeros(pop_size)
for j in range(pop_size):
    pop[:,j] = np.random.randint(0,2,ind_size).astype(int)
    fitness[j] = knapsack_value_penalty(pop[:,j], values, weights, max_weight)
######

# initialize stats and output
best_fitness = min(fitness)
stats[0,:] = np.array([0,best_fitness, best_fitness])
print('Iteration | Best this iter |    Best ever')

for iter in range(num_iter):
    # tournament selection
    sorted_pos = fitness.argsort() # sort pop by increasing fitness
    fitness = fitness[sorted_pos]
    pop = pop[:,sorted_pos]
    select_pop = np.zeros((ind_size,pop_size)).astype(int) # initialize selected population
    for j in range(pop_size):
        subset_pos = np.random.choice(pop_size,tourn_size,replace=False) # select without replacement
        # print(subset_pos)
        smallest_pos = np.min(subset_pos) # choose index corresponding to lowest fitness
        select_pop[:,j] = pop[:,smallest_pos]

    # one-point crossover (mating)
    cx_pop = np.zeros((ind_size,pop_size)).astype(int) # initialize crossover population
    for j in range(int(pop_size/2)):  # pop_size must be even
        #######
        parent1, parent2 = select_pop[:,2*j], select_pop[:,2*j+1]
        child1, child2 = parent1.copy(), parent2.copy()
        if np.random.uniform() < cx_prob: # crossover occurs
            cx_point = np.random.randint(1,ind_size) # crossover point between 0 and ind_size-2
            child1[0:cx_point], child2[0:cx_point] = parent2[0:cx_point], parent1[0:cx_point]
        cx_pop[:,2*j] = child1
        cx_pop[:,2*j+1] = child2
        ######

    # gaussian mutation (rewritten to remove nested loop for speed)
    mut_pop = np.zeros((ind_size,pop_size)).astype(int) # initialize mutation population
    for j in range(pop_size):
        individual = cx_pop[:,j].copy() # copy is necessary to avoid conflicts in memory
        if np.random.uniform()<mut_prob:
            ###### swap the ith entry with a randomly selected entry with prob ind_prob
            # For mutation of permutation variables it is common to use Shuffling Indices. To do just make a copy of the individual then loop over each variable and with
            #  probability ind_prob swap it with another randomly selected variable in the individual. It's possible that you may end up swapping a variable with itself, but that's OK.
            # To initialize you'll to use a loop since it's only possible to create one random permutation at a time.
            for i in range(individual.size):
                if np.random.uniform()<ind_prob:
                    if individual[i] == 1 :
                        individual[i] = 0
                    else:
                        individual[i] = 1
            ######
        mut_pop[:,j] = individual.copy() # copy is necessary to avoid conflicts in memory

    # fitness evaluation with local search
    pop = mut_pop.copy()
    for j in range(pop_size):
        fitness[j] = knapsack_value_penalty(pop[:,j], values, weights, max_weight)

    # collect stats and output to screen
    min_fitness = min(fitness) # best for this iteration
    if min_fitness < best_fitness: # best for all iterations
        best_fitness = min_fitness
        index = np.argmin(fitness)
        best_x = pop[:,index]

    stats[iter+1,:] = np.array([iter+1,min_fitness, best_fitness])
    if (iter+1) % update_iter == 0:
        print(f"{stats[iter+1,0]:9.0f} | {stats[iter+1,1]:14.3e} | {stats[iter+1,2]:12.3e}")

######
print(f"The minimum value found of the fitnexx function is {best_fitness:.0f}")
print("The tour that is minimum is:")
######
print('(',', '.join(f"{x}" for x in best_x),')')
