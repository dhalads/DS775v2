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
df = pd.read_csv('../../Lessons/Lesson 10 Presentation - RecSys 1/data/ted-talks/ted_main.csv')

#print the shape of the dataframe
print(f"The shape is {df.shape}")

#get the column info
df.info()

df1 = df[df['duration'] >= 5*60]
print(f"shape is {df1.shape}")
df2 = df[df['num_speaker'] == 1]
print(f"shape is {df2.shape}")
views90 = df['views'].quantile(0.1)
df3 = df[df['views'] >= views90]
print(f"shape is {df3.shape}")

df['score']= df['comments']/df['views']/1000
df4 = df.sort_values("score", ascending = False)
display(df4[['description','main_speaker', 'views']].head(10))

# #Convert release_date into pandas datetime format
df['published_year'] = pd.to_datetime(df['published_date'],unit = 's', errors='coerce')

# #Extract year from the datetime and convert to integer. (Again, we're chaining functions)
df['year'] = df['published_year'].apply(lambda x: x.year if x != np.nan else np.nan)
df['ratings'] =  df['ratings'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


def uniqueRatings(df):
    #in steps
    #convert the genres list in a series of columns
    # step1 = df.apply(lambda x:pd.Series(x['production_companies']),axis=1)
    # print(f"Step 1\n{step1}")

    # #this step converts the rows into columns and "stacks" them all together
    # step2 = step1.stack()
    # print(f"Step 2\n{step2}")

    # #let's get just the unique values from this series
    # step3 = step2.unique()
    # print(f"Step 3\n{step3}")
    # print(f"Step 3 is a \n{type(step3)}")

    # #numpy arrays can be joined just like lists, so let's join it to create a comma-delimited string
    # step4 = ', '.join(step3)
    # print(f"Step 4\n{step4}")


    # #let's do it all in one step
    allRatings = ', '.join(df.apply(lambda x:pd.Series(x['ratings']),axis=1).stack().unique())
    return(allRatings)




def build_chart(df, percentile=0.9):

    #Ask for preferred genres
    print(uniqueRatings(df))
    print("Input preferred rating:")
    rating = input()

    #Ask for lower limit of duration
    print("Input earliest published year(between 2006 and 2017):")
    low_year = int(input())

    #Ask for upper limit of duration
    print("Input latest published year(between 2006 and 2017):")
    high_year = int(input())

    #Define a new movies variable to store the preferred movies. Copy the contents of gen_df to movies
    movies = df.copy()

    #Filter based on the condition
    movies = movies[(movies['ratings'].apply(lambda x: rating in x)) & #updated filtering based on a list.
                    (movies['year'] >= low_year) &
                    (movies['year'] <= high_year)]

    #Compute the values of C and m for the filtered movies
    # C = movies['vote_average'].mean()
    m = movies['views'].quantile(percentile)

    #Only consider movies that have higher than m votes. Save this in a new dataframe q_movies
    q_movies = movies.copy().loc[movies['views'] >= m]

    #Calculate score using the IMDB formula
    # q_movies['score'] = q_movies[]
    q_movies['score']= q_movies['comments']/q_movies['views']/1000

    #Sort movies in descending order of their scores
    q_movies = q_movies.sort_values('score', ascending=False)

    return q_movies

out_movies = build_chart(df, .9)
# display(out_movies.head)
display(out_movies[['name','main_speaker', 'published_year']].head(10))