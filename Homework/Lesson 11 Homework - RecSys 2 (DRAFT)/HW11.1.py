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

import os

#read in the data
df = pd.read_csv('data/restaurant/rating_final.csv')
df['rating'] = df['rating'] + df['food_rating'] + df['service_rating'] + 1
df = df.drop(columns=["food_rating", "service_rating"])
df.info()
print(df.describe())
print(df.shape)
print(df.head())

X = df.copy()
y = df['userID']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=None, random_state=42)

#first determine the median of our ratings (we could have done this by hand, but numpy does it so well... )
print(f"The median of this rating range is {np.median(df['rating'])}")

#define a baseline model to always return the median
def baseline(user_id, place_id, *args):
    return 5.0

#Function to compute the RMSE score obtained on the testing set by a model
def score(cf_model, X_test, *args):

    #Construct a list of user-place tuples from the testing dataset
    id_pairs = zip(X_test['userID'], X_test['placeID'])

    #Predict the rating for every user-movie tuple
    y_pred = np.array([cf_model(user, place, *args) for (user, place) in id_pairs])

    #Extract the actual ratings given by the users in the test data
    y_true = np.array(X_test['rating'])

    #Return the final RMSE score
    return mean_squared_error(y_true, y_pred, squared=False)

#let's test it with our baseline model
print(f"baseline score = {score(baseline, X_test)}")


#Build the ratings matrix using pivot_table function
r_matrix = X_train.pivot_table(values='rating', index='userID', columns='placeID')

print(r_matrix.head())

#Create a dummy ratings matrix with all null values imputed to 0
r_matrix_dummy = r_matrix.copy().fillna(0)
# Import cosine_score
# from sklearn.metrics.pairwise import cosine_similarity

#Compute the cosine similarity matrix using the dummy ratings matrix
cosine_sim = cosine_similarity(r_matrix_dummy, r_matrix_dummy)

#Convert into pandas dataframe
cosine_sim = pd.DataFrame(cosine_sim, index=r_matrix.index, columns=r_matrix.index)

print(cosine_sim.head(10))

#User Based Collaborative Filter using Weighted Mean Ratings
def cf_user_wmean(user_id, place_id, ratings_matrix, c_sim_matrix):

    #Check if movie_id exists in r_matrix
    if place_id in ratings_matrix:
        
        #Get the similarity scores for the user in question with every other user
        sim_scores = c_sim_matrix[user_id]

        #Get the user ratings for the movie in question
        m_ratings = ratings_matrix[place_id]

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
        wmean_rating = 5.0

    return wmean_rating



print(f"Weighted Mean score = {score(cf_user_wmean, X_test, r_matrix_dummy, cosine_sim)}")


