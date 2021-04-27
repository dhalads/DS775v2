# computational imports
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Returns the list top 3 elements or entire list; whichever is more.
def generate_list(x):
    if isinstance(x, list):
        names = [ele['name'] for ele in x]
        #Check if more than 3 elements exist. If yes, return only first three.
        #If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names
    #Return empty list in case of missing/malformed data
    return []

f = lambda x: x[:3]
print(f([1,2]))
print(f([1,2,3]))
print(f([1,2,3,4]))

#read in the data
df = pd.read_csv('data/tmdb_5000_movies.csv', encoding = "ISO-8859-1", usecols=("title", "genres", "keywords", "overview", "runtime", "budget", "production_companies", "vote_count", "vote_average"))
df['genres'] =  df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['keywords'] =  df['keywords'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []).apply(lambda x: x[:3])
df['production_companies'] =  df['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []).apply(lambda x: x[:3])
df['overview'] = df['overview'].fillna('')
#Function that creates a soup out of the desired metadata
def create_soup(x):
    return ' '.join(x['genres']) + ' ' + ' '.join(x['keywords']) + ' ' + ' '.join(x['production_companies']) + ' ' + x['overview']

#create a column with the soup in it    
df['soup'] = df.apply(create_soup, axis=1)


print(f'The soup for Toy Story is: \n{df["soup"][0]}')


#Apply the generate_list function to cast and keywords
# df['cast'] = df['cast'].apply(generate_list)
# df['keywords'] = df['keywords'].apply(generate_list)
#Only consider a maximum of 3 genres
# df['genres'] = df['genres'].apply(lambda x: x[:3])

#Import TfIdfVectorizer from the scikit-learn library
# from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = CountVectorizer(stop_words='english')
vectorizer_matrix = vectorizer.fit_transform(df['soup'])
print(vectorizer_matrix.shape)

# #this extracts all the words (features) in the matrix - we'll use this for our columns
# feature_names = tfidf.get_feature_names()
# #this extracts the IDs of the movies - we'll use this for our rows
# corpus_index = snip['id']
# #this puts both into a dataframe.
# #The tfidf_matrix is usually a sparse matrix, meaning not all row/col combinations have a value. Using todense() puts a zero in that row/col slot

# pd.DataFrame(tfidf_matrix.todense(), index=corpus_index, columns=feature_names)

# Import linear_kernel to compute the dot product
# from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(vectorizer_matrix, vectorizer_matrix)

#let's look at what we've got.
display(pd.DataFrame(cosine_sim, columns=df['title'], index=df['title']))

#create the reverse mapping
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
#print it 
print(f'The index series looks like this: \n{indices}')

#if I wanted to get the index from the title I would do this:
print(f'The index for Waiting to Exhale is: {indices["Halloween"]}')
print(cosine_sim[2])
sim_scores = list(enumerate(cosine_sim[2]))
print(sim_scores)
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
print(sim_scores)

top_two = [i[0] for i in sim_scores[0:2]]
print(f'The top two indices are: {top_two}')

display(df.iloc[top_two])

def content_recommender(df, title, cosine_sim, indices, topN=2):
    # Obtain the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie and convert to tuples
    sim_scores = list(enumerate(cosine_sim[idx]))
    #delete the movie that was passed in
    del sim_scores[idx]
    
    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top-n most similar movies.
    sim_scores = sim_scores[:topN]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return df[['title', 'genres']].iloc[movie_indices]

out_recommendation = content_recommender(df, 'Halloween', cosine_sim, indices, 10)
display(out_recommendation)