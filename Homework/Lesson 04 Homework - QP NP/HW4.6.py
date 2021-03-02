import numpy as np
import pandas as pd

# load the data + random assignment
num_districts = 10
min_voters_in_district = 150
max_voters_in_district = 350

dems = [152,81,75,34,62,38,48,74,98,66,83,86,72,28,112,45,93,72]
reps = [62,59,83,52,87,87,69,49,62,72,75,82,83,53,98,82,68,98]
cities = pd.DataFrame( data = {'dems':dems, 'reps':reps})

# initial assignment
assign = np.random.randint(low=0,high=num_districts,size = 18)

# ***maximize*** this function
def fitness_districts(assign, cities, num_districts):
    df = cities.groupby(assign).sum()
    fitness = sum(df['reps'] > df['dems'])
    total_voters = np.zeros(num_districts, dtype=np.int32)
    total_voters[df.index] = df.sum(axis=1)
    s1 = np.maximum(total_voters, 150)
    s2 = np.minimum(s1, 350)
    s3 = np.abs(s2 - total_voters)
    fitness -= s3.sum()
    # fitness -= np.abs(
    #     np.minimum(np.maximum(total_voters, 150), 350) - total_voters).sum()
    return -(fitness)

# to display output, not used in optimization
def summarize_districts(assign, cities):
    reps = np.zeros(num_districts, dtype=np.int32)
    dems = np.zeros(num_districts, dtype=np.int32)
    df = cities.groupby(assign).sum()
    reps[df.index] = df['reps']
    dems[df.index] = df['dems']
    total = reps + dems
    delta = np.minimum(np.maximum(total, min_voters_in_district),
                       max_voters_in_district) - total
    rep_win = reps > dems
    dict = {'reps': reps, 'dems': dems, 'total': total, 'rep_win': rep_win}
    return (pd.DataFrame(data=dict))

def move_one_city(assign, num_districts):
    num_cities = assign.shape[0] # or len(assign)
    new_assign = assign.copy()
    switch_city = np.random.randint(num_cities) # which city to assign new random district
    while new_assign[switch_city] == assign[switch_city]: # loops until new and old are different
        new_assign[ switch_city] = np.random.randint(num_districts)
    return new_assign


from locsearch import LocalSearcher

class Gerrymanding(LocalSearcher):
    """
    Test local search with a traveling salesman problem
    """
    
    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, cities, num_districts):
        self.cities = cities
        self.num_districts = num_districts
        super(Gerrymanding, self).__init__(state)  # important!

    def move(self):
        self.state = move_one_city(self.state, self.num_districts)

    def objective(self):
        return fitness_districts(self.state,self.cities, self.num_districts)


print(summarize_districts(assign, cities))
print(f"best_f={fitness_districts(assign, cities, num_districts)}")
gm = Gerrymanding(assign, cities, num_districts)


# uncomment to override default search and output settings
gm.max_no_improve = 300
gm.update_iter = 500

# call the local search method in our object to do the search
best_assign, best_f = gm.localsearch()

print(summarize_districts(best_assign, cities))
print(f"best_f={best_f}")