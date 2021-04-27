# EXECUTE FIRST

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
HTML(response.read().decode("utf-8"));
import os


#read in the data
df = pd.read_csv('../../Lessons/Lesson 10 Presentation - RecSys 1/data/movies_metadata.csv')

#print the shape of the dataframe
print(f"The shape is {df.shape}")

#get the column info
df.info()

#####################
# Helper Functions
#####################
#converts ints & string representations of numbers to floats
def to_float(x):
    try:
        x = float(x)
    except:
        x = np.nan
    return x

#Helper function to convert NaT to 0 and all other years to integers.
def convert_int(x):
    try:
        return int(x)
    except:
        return 0

#we can run both apply and astype in one line by chaining them
df['budget'] = df['budget'].apply(to_float).astype('float')

#Convert release_date into pandas datetime format
df['release_date'] = pd.to_datetime(df['release_date'],errors='coerce')

#Extract year from the datetime and convert to integer. (Again, we're chaining functions)
df['year'] = df['release_date'].apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan).apply(convert_int)

#convert vote_count to integer
df['vote_count'] = df['vote_count'].apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan).apply(convert_int)

#Convert all NaN into stringified empty lists and apply literal eval and convert to list by chaining functions
df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

#filter to just the relevant columns
df = df[['id','title','budget', 'genres', 'overview', 'revenue', 'runtime', 'vote_average', 'vote_count', 'year']]
display(df.head())
# display(df)

#let's fetch just the first 5 rows of our dataframe
snip = df[:5]
display(snip)

#let's create a filter that will be True if "Family" is in the list of genres for each movie
hasFamilyFilter = snip['genres'].apply(lambda x: "Family" in x)
print(f'Family filter values \n {hasFamilyFilter}')

#let's create a filter that will be True if "Drama" is in the list of genres of each movie
hasDramaFilter = snip['genres'].apply(lambda x: "Drama" in x)
print(f'Drama filter values \n{hasDramaFilter}')

#let's filter our dataset to just those movies that have Family OR Drama. Note the placement of the parenthesis
display(snip[(hasFamilyFilter) | (hasDramaFilter)])

#let's filter our dataset to just those movies that have Comedy AND Romance OR have a vote_count > 5000.
#let's use variables for our two genres
selected1 = 'Romance'
selected2 = 'Comedy'

#instead of creating stand-alone filters, we'll filter "on the fly" using the apply right in the filter
#again, pay attention to where the parentheses go
snip[(snip['vote_count'] > 5000) |
     ((snip['genres'].apply(lambda x: selected1 in x)) &
      (snip['genres'].apply(lambda x: selected2 in x)))]

#in steps
#convert the genres list in a series of columns
step1 = snip.apply(lambda x:pd.Series(x['genres']),axis=1)
print(f"Step 1\n{step1}")

#this step converts the rows into columns and "stacks" them all together
step2 = step1.stack()
print(f"Step 2\n{step2}")

#let's get just the unique values from this series
step3 = step2.unique()
print(f"Step 3\n{step3}")
print(f"Step 3 is a \n{type(step3)}")

#numpy arrays can be joined just like lists, so let's join it to create a comma-delimited string
step4 = ', '.join(step3)
print(f"Step 4\n{step4}")


#let's do it all in one step
allGenres = ', '.join(snip.apply(lambda x:pd.Series(x['genres']),axis=1).stack().unique())
allGenres

#fetch C from the whole dataset
C = df['vote_average'].mean()
print(f"C is {C}")

#fetch m from the whole dataset
m = df['vote_count'].quantile(.8)
print(f"m is {m}")

#filter to movies that have greater than or equal to 80% of the votes
df = df[df['vote_count'] >= m]

#see how many movies are left.
df.shape

def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    # Compute the weighted score
    return (v/(v+m) * R) + (m/(m+v) * C)

df['score1'] = df.apply(weighted_rating, args=(m,C), axis=1)
display(df.head())

#fetch c from the already filtered data
C2 = df['vote_average'].mean()
print(f"C is {C2}")

df['score2'] = df.apply(weighted_rating, args=(m,C2), axis=1)
display(df.head())

def build_chart(gen_df, percentile=0.8):

    #Ask for preferred genres
    print("Input preferred genre")
    genre = input()

    #Ask for lower limit of duration
    print("Input shortest duration")
    low_time = int(input())

    #Ask for upper limit of duration
    print("Input longest duration")
    high_time = int(input())

    #Ask for lower limit of timeline
    print("Input earliest year")
    low_year = int(input())

    #Ask for upper limit of timeline
    print("Input latest year")
    high_year = int(input())

    #Define a new movies variable to store the preferred movies. Copy the contents of gen_df to movies
    movies = gen_df.copy()

    #Filter based on the condition
    movies = movies[(movies['genres'].apply(lambda x: genre in x)) & #updated filtering based on a list.
                    (movies['runtime'] >= low_time) &
                    (movies['runtime'] <= high_time) &
                    (movies['year'] >= low_year) &
                    (movies['year'] <= high_year)]

    #Compute the values of C and m for the filtered movies
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(percentile)

    #Only consider movies that have higher than m votes. Save this in a new dataframe q_movies
    q_movies = movies.copy().loc[movies['vote_count'] >= m]

    #Calculate score using the IMDB formula
    q_movies['score'] = q_movies.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average'])
                                       + (m/(m+x['vote_count']) * C)
                                       ,axis=1)

    #Sort movies in descending order of their scores
    q_movies = q_movies.sort_values('score', ascending=False)

    return q_movies

out_movies = build_chart(df, .8)
# display(out_movies.head)

#set the score rank column
out_movies['scoreRank'] = np.arange(len(out_movies))
#sort by score1 and set the score1rank column
out_movies = out_movies.sort_values('score1', ascending=False)
out_movies['score1Rank'] = np.arange(len(out_movies))
#sort by score2 and set the score2rank column
out_movies = out_movies.sort_values('score2', ascending=False)
out_movies['score2Rank'] = np.arange(len(out_movies))
#resort by score
out_movies = out_movies.sort_values('score', ascending=False)

#display the final result with just name and scores
display(out_movies[['title','score1Rank', 'score2Rank', 'scoreRank' ]])