import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from scipy.optimize import minimize
import json
from simanneal import Annealer

#####
pop_size = 100 # should be even due to the way we'll implement crossover
ind_size = 48 # determines number of input variables for each tour
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

stats = np.zeros((num_iter+1,3)) # for collecting statistics

with open("data/Caps48.json", "r") as tsp_data:
    tsp = json.load(tsp_data)
distance_matrix = tsp["DistanceMatrix"]
optimal_tour = tsp["OptTour"]
opt_dist = tsp["OptDistance"]/1000 # converted to kilometers
xy = np.array(tsp["Coordinates"])

######
# objective or fitness function
def tour_distance(tour, dist_mat):
    distance = dist_mat[tour[-1]][tour[0]]
    for gene1, gene2 in zip(tour[0:-1], tour[1:]):
        distance += dist_mat[gene1][gene2]
    return distance
######

###### may need to also enforce that the variables are integers and not floats
#initialize population and fitness np.random.randint(0,2,size=20,dtype=bool)
# init_tour = np.random.permutation(np.arange(len(distance_matrix))).astype(int).tolist()
# pop = np.random.uniform(low=lower, high=upper, size = (ind_size,pop_size))
pop = np.zeros((ind_size, pop_size)).astype(int)
fitness = np.zeros(pop_size)
for j in range(pop_size):
    pop[:,j] = np.random.permutation(ind_size).astype(int)
    fitness[j] = tour_distance(pop[:,j], distance_matrix)
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
        # if np.random.uniform() < cx_prob: # crossover occurs
        #     cx_point = np.random.randint(1,ind_size) # crossover point between 0 and ind_size-2
        #     child1[0:cx_point], child2[0:cx_point] = parent2[0:cx_point], parent1[0:cx_point]
        swap_idx_start = np.random.randint(ind_size)
        swap_size =  np.random.randint(0, ind_size - swap_idx_start)
        hole = np.full( ind_size, False, dtype = bool)
        hole[swap_idx_start:swap_idx_start + swap_size] = True

        child1[~hole] = np.array([x for x in parent2 if x not in parent1[hole]])
        child2[~hole] = np.array([x for x in parent1 if x not in parent2[hole]])
        # print(hole)
        # print(parent1[hole])
        # print(parent2[hole])
        # print(parent1,parent2)
        # print(child1,child2)
        cx_pop[:,2*j] = child1
        cx_pop[:,2*j+1] = child2
        ######

    # gaussian mutation (rewritten to remove nested loop for speed)
    mut_pop = np.zeros((ind_size,pop_size)).astype(int) # initialize mutation population
    for j in range(pop_size):
        individual = cx_pop[:,j].copy() # copy is necessary to avoid conflicts in memory
        if np.random.uniform()<mut_prob:
            ###### swap the ith entry with a randomly selected entry with prob ind_prob
            swap_idx_1 = np.random.randint(48)
            swap_idx_2 =  np.random.randint(48)
            swap_value =  individual[swap_idx_1]
            individual[swap_idx_1] = individual[swap_idx_2]
            individual[swap_idx_2] = swap_value
            ######
        mut_pop[:,j] = individual.copy() # copy is necessary to avoid conflicts in memory

    # fitness evaluation with local search
    pop = mut_pop.copy()
    for j in range(pop_size):
        fitness[j] =tour_distance(pop[:,j], distance_matrix)

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
print(f"The minimum value found of the fitnexx function is {best_fitness/1000:.0f}")
print("The tour that is minimum is:")
######
print('(',', '.join(f"{x}" for x in best_x),')')