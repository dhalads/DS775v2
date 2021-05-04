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
from surprise.model_selection import cross_validate

np.random.seed(1)

def create_predictors():
    #read in the data
    df = pd.read_csv('data/restaurant/rating_final.csv')
    df['rating'] = df['rating'] + df['food_rating'] + df['service_rating'] + 1
    df = df.drop(columns=["food_rating", "service_rating"])
    df.info()
    print(df.describe())
    print(df.shape)
    print(df.head())

    #The Reader object helps in parsing the file or dataframe containing ratings
    reader = Reader(rating_scale=(1,7)) # defaults to (0,5)

    #Create the dataset to be used for building the filter
    data = Dataset.load_from_df(df, reader)

    #Define the algorithm object; in this case kNN
    np.random.seed(1)
    knn_predictor =KNNBasic(k = 21, verbose=False)
    cross_validate(knn_predictor,data,cv=5,verbose=True)

    svd_predictor = SVD()
    cross_validate(svd_predictor,data,cv=5,verbose=True)
    return((data, knn_predictor, svd_predictor))

def create_soup(x):
    cols = ['price', 'dress_code', 'accessibility', 'Rambience', 'alcohol', 'smoking_area']
    output = ''
    for col in cols:
        output = output + ' ' + (x[col].replace(' ', '').lower())
    # output = output + ' ' + "happy Coke".replace(' ', '').lower()
    output = output.strip()
    return(output)

def create_Cosine_matrix():
    df = pd.read_csv('data/restaurant/geoplaces2.csv')
    #create a column with the soup in it
    df['soup'] = df.apply(create_soup, axis=1)

    vectorizer = CountVectorizer(stop_words='english')
    vectorizer_matrix = vectorizer.fit_transform(df['soup'])
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(vectorizer_matrix, vectorizer_matrix)
    # create the reverse mapping
    indices = pd.Series(df.index, index=df['name']).drop_duplicates()

    return((df, cosine_sim, indices))


def hybrid(user_id, name, cosine_matrix, indices, topN, data, predictor):

    # Obtain the index of the movie that matches the title
    idx = indices[name]
    # Get the pairwsie similarity scores of all movies with that movie and convert to tuples
    sim_scores = list(enumerate(cosine_matrix[idx]))
    #delete the movie that was passed in
    del sim_scores[idx]

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top-n most similar movies.
    sim_scores = sim_scores[:topN]

    # Get the place indices
    place_indices = [i[0] for i in sim_scores]

    #Extract the metadata of the aforementioned items
    items = data.iloc[place_indices][['placeID']]
    
    #Compute the predicted ratings using the SVD filter
    items['pred_rating'] = items['placeID'].apply(lambda x: predictor.predict(user_id, x).est)

    #Sort the items in decreasing order of predicted rating
    items = items.sort_values('pred_rating', ascending=False)

    output = pd.merge(items, data, on = "placeID", how = "inner")
    #Return the top 10 items as recommendations
    return output[['placeID', 'name', 'price', 'dress_code', 'accessibility', 'Rambience', 'alcohol', 'smoking_area', 'pred_rating']].head(10)



cos_result = create_Cosine_matrix()
display(cos_result)
pred_result = create_predictors()

user_id='U1077'
place_name = 'cafe ambar'
print(f"user={user_id}, name={place_name}, predictor=KNN")
results1 = hybrid(user_id, place_name, cos_result[1], cos_result[2], 25, cos_result[0], pred_result[1])
display(results1)
print(f"user={user_id}, name={place_name}, predictor=SVD")
results2 = hybrid(user_id, place_name, cos_result[1], cos_result[2], 25, cos_result[0], pred_result[2])
display(results2)
user_id='U1065'
place_name = 'cafe ambar'
print(f"user={user_id}, name={place_name}, predictor=KNN")
results1 = hybrid(user_id, place_name, cos_result[1], cos_result[2], 25, cos_result[0], pred_result[1])
display(results1)
print(f"user={user_id}, name={place_name}, predictor=SVD")
results2 = hybrid(user_id, place_name, cos_result[1], cos_result[2], 25, cos_result[0], pred_result[2])
display(results2)