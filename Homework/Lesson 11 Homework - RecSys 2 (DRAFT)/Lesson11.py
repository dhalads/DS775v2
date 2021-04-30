# EXECUTE FIRST

# computational imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, KNNBasic, SVD

from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
# for reading files from urls
import urllib.request
# display imports
from IPython.display import display, IFrame
from IPython.core.display import HTML

# import notebook styling for tables and width etc.
response = urllib.request.urlopen('https://raw.githubusercontent.com/DataScienceUWL/DS775v2/master/ds755.css')
HTML(response.read().decode("utf-8"))

# import pandas as pd
# import numpy as np

#load the information about users
users = pd.DataFrame({'user_id': [1,2,3,4,5],
                     'age': [24,53,23,20,55],
                     'sex': ['M','F','M','F','M'],
                     'occupation': ['technician', 'writer','teacher','technician','teacher'],
                     'zip_code': ['90210', '53704', '53706','53704','90210']})

display(users.head())

movies = pd.DataFrame({'movie_id': [1,2,3,4,5],
                      'title':['Toy Story','Titanic','Star Wars: The Clone Wars', 'Gone with the Wind', 'Sharknado']})


display(movies.head())

#generate a rating for each user/movie combination
ratings = pd.DataFrame(np.array(np.meshgrid([1, 2, 3,4,5], [1,2,3,4,5])).T.reshape(-1,2), columns=['user_id', 'movie_id'])
np.random.seed(1)
randratings = np.random.randint(1,6, ratings.shape[0])

ratings['rating'] = randratings

#we have 5 * 5 or 25 rows of data in the ratings, but we'll just look at the first 10
ratings.head(10)

#Import the train_test_split function
# from sklearn.model_selection import train_test_split

#Assign X as the original ratings dataframe and y as the user_id column of ratings.
X = ratings.copy()
y = ratings['user_id']

#Split into training and test datasets, stratified along user_id
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state=42)

#compare X_train to X_test
display(X_train)
display(X_test)

#Import the mean_squared_error function
# from sklearn.metrics import mean_squared_error

#test data
test_y_true = [3, -0.5, 2, 7]
test_y_pred = [2.5, 0.0, 2, 8]

#this returns MSE (not what we want)
print(mean_squared_error(test_y_true, test_y_pred))

#this returns the root mean squared error (and is what we want to use)
mean_squared_error(test_y_true, test_y_pred, squared=False)

#first determine the median of our ratings (we could have done this by hand, but numpy does it so well... )
print(f"The median of this rating range is {np.median(np.arange(np.min(ratings['rating']), (np.max(ratings['rating']) + 1)))}")

#define a baseline model to always return the median
def baseline(user_id, movie_id, *args):
    return 3.0

#Function to compute the RMSE score obtained on the testing set by a model
def score(cf_model, X_test, *args):

    #Construct a list of user-movie tuples from the testing dataset
    id_pairs = zip(X_test['user_id'], X_test['movie_id'])

    #Predict the rating for every user-movie tuple
    y_pred = np.array([cf_model(user, movie, *args) for (user, movie) in id_pairs])

    #Extract the actual ratings given by the users in the test data
    y_true = np.array(X_test['rating'])

    #Return the final RMSE score
    return mean_squared_error(y_true, y_pred, squared=False)

#let's test it with our baseline model
score(baseline, X_test)

#Build the ratings matrix using pivot_table function
r_matrix = X_train.pivot_table(values='rating', index='user_id', columns='movie_id')

r_matrix.head()

#User Based Collaborative Filter using Mean Ratings
def cf_user_mean(user_id, movie_id, ratings_matrix):
    
    #Check if movie_id exists in r_matrix (rm)
    if movie_id in ratings_matrix:
        #Compute the mean of all the ratings given to the movie
        mean_rating = ratings_matrix[movie_id].mean()
    
    else:
        #Default to a rating of 3.0 in the absence of any information
        mean_rating = 3.0

    return mean_rating

score(cf_user_mean, X_test, r_matrix)

#Create a dummy ratings matrix with all null values imputed to 0
r_matrix_dummy = r_matrix.copy().fillna(0)
# Import cosine_score
# from sklearn.metrics.pairwise import cosine_similarity

#Compute the cosine similarity matrix using the dummy ratings matrix
cosine_sim = cosine_similarity(r_matrix_dummy, r_matrix_dummy)

#Convert into pandas dataframe
cosine_sim = pd.DataFrame(cosine_sim, index=r_matrix.index, columns=r_matrix.index)

cosine_sim.head(10)

#User Based Collaborative Filter using Weighted Mean Ratings
def cf_user_wmean(user_id, movie_id, ratings_matrix, c_sim_matrix):

    #Check if movie_id exists in r_matrix
    if movie_id in ratings_matrix:
        
        #Get the similarity scores for the user in question with every other user
        sim_scores = c_sim_matrix[user_id]

        #Get the user ratings for the movie in question
        m_ratings = ratings_matrix[movie_id]

        #Extract the indices containing NaN in the m_ratings series
        idx = m_ratings[m_ratings.isnull()].index

        #Drop the NaN values from the m_ratings Series
        m_ratings = m_ratings.dropna()
        
        #Drop the corresponding cosine scores from the sim_scores series
        sim_scores = sim_scores.drop(idx)

        #Compute the final weighted mean
        wmean_rating = np.dot(sim_scores, m_ratings)/ sim_scores.sum()

    else:
        #Default to a rating of 3.0 in the absence of any information
        wmean_rating = 3.0

    return wmean_rating



score(cf_user_wmean, X_test, r_matrix, cosine_sim)

#merge the training set with the user data
merged_df = pd.merge(X_train, users.copy())

#Compute the mean rating of every movie by gender
gender_mean = merged_df[['movie_id', 'sex', 'rating']].copy().groupby(['movie_id', 'sex'])['rating'].mean()

display(gender_mean.head())

#Set the index of the users dataframe to the user_id
#we need to do this so that we can fetch the right data in our model function
users = users.set_index('user_id')

display(users)

 #Gender Based Collaborative Filter using Mean Ratings
def cf_gender(user_id, movie_id, ratings_matrix, user_df, gen_mean):
    
    #Check if movie_id exists in r_matrix (or training set)
    if movie_id in ratings_matrix:
        #Identify the gender of the user
        gender = user_df.loc[user_id]['sex']
        
        #Check if the gender has rated the movie
        if gender in gen_mean[movie_id]:
            
            #Compute the mean rating given by that gender to the movie
            gender_rating = gen_mean[movie_id][gender]
        
        else:
            gender_rating = 3.0
    
    else:
        #Default to a rating of 3.0 in the absence of any information
        gender_rating = 3.0
    
    return gender_rating

score(cf_gender,  X_test, r_matrix, users, gender_mean)

#Compute the mean rating by gender and occupation
gen_occ_mean = merged_df[['sex', 'rating', 'movie_id', 'occupation']].pivot_table(
    values='rating', index='movie_id', columns=['occupation', 'sex'], aggfunc='mean')

gen_occ_mean.head()

#Gender and Occupation Based Collaborative Filter using Mean Ratings
def cf_gen_occ(user_id, movie_id, user_df, gen_occ_mean_df):
    
    #Check if movie_id exists in gen_occ_mean
    if movie_id in gen_occ_mean_df.index:
        
        #Identify the user
        user = user_df.loc[user_id]
        
        #Identify the gender and occupation
        gender = user['sex']
        occ = user['occupation']
        
        #Check if the occupation has rated the movie
        if occ in gen_occ_mean_df.loc[movie_id]:
            
            #Check if the gender has rated the movie
            if gender in gen_occ_mean_df.loc[movie_id][occ]:
                
                #Extract the required rating
                rating = gen_occ_mean_df.loc[movie_id][occ][gender]
                
                #Default to 3.0 if the rating is null
                if np.isnan(rating):
                    rating = 3.0
                
                return rating
            
    #Return the default rating    
    return 3.0

#compute the RMSE
score(cf_gen_occ,  X_test, users, gender_mean)

# this has been edited - replace "evaluate" with "cross_validate"

#Import the required classes and methods from the surprise library
# from surprise import Reader, Dataset, KNNBasic

#Define a Reader object
#The Reader object helps in parsing the file or dataframe containing ratings
reader = Reader(rating_scale=(1,5)) # defaults to (0,5)

#Create the dataset to be used for building the filter
data = Dataset.load_from_df(ratings, reader)

#define a random seed for consistent results
np.random.seed(1)
#Define the algorithm object; in this case kNN
knn = KNNBasic(k=3, verbose=False) #the default for k is 40, we're also setting verbose to False to supress messages

#This code cross validates (evaluates) the model
from surprise.model_selection import cross_validate
knn_cv = cross_validate(knn, data, measures=['RMSE'], cv=5, verbose=True)
print(knn_cv)

#to extract the mean RMSE, we need to get the mean of the test_rmse values
knn_RMSE = np.mean(knn_cv['test_rmse'])
print(f'\nThe RMSE across five folds was {knn_RMSE}')

# this has been edited - replace "evaluate" with "cross_validate"
#Import SVD
# from surprise import SVD

#define a random seed for consistent results
np.random.seed(1)
#Define the SVD algorithm object
svd = SVD()

#Evaluate the performance in terms of RMSE
svd_cv = cross_validate(svd, data, measures=['RMSE'], cv=5, verbose=True)
#to extract the mean RMSE, we need to get the mean of the test_rmse values
svd_RMSE = np.mean(svd_cv['test_rmse'])
print(f'\nThe RMSE across five folds was {svd_RMSE}')
