import pandas as pd
import numpy as np

desired_width = 320

# This just for viewing the dataframe in the o/p
pd.set_option('display.width', desired_width)

np.set_printoptions(linewidth=desired_width)

pd.set_option('display.max_columns', 12)

r_cols = ['user_id', 'movie_id', 'rating']  # creating three cols for the dataframe for the actually user feedback
ratings = pd.read_csv("C:\\Users\\Shreyas\\MLCourse\\ml-100k\\u.data", sep="\t", names=r_cols, usecols=range(3))

m_cols = ['movie_id', 'title']  # creating 2 cols for the mobie id and title
movies = pd.read_csv("C:\\Users\\Shreyas\\MLCourse\\ml-100k\\u.item", sep='|', names=m_cols, usecols=range(2),
                     encoding='latin-1')  # there was an encoding problem in the new version of pandas for "|".

ratings = pd.merge(ratings, movies)

movieRatings = ratings.pivot_table(index=['user_id'], columns=['title'],
                                   values="rating")  # Create a spreadsheet-style pivot table as a DataFrame.
# print(movieRatings.head())
SelectMovie = input(str("Enter the movie you want recommendation,similar movies,rating for!!"))
Selector_movie = movieRatings[SelectMovie]
print(Selector_movie)

similarMovies = movieRatings.corrwith(Selector_movie)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
# print(df)

similarMovies = similarMovies.sort_values(ascending=False)
# print(similarMovies)


movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})#Aggregate using one or more operations over the specified axis.
# print(movieStats)

popularMovies = movieStats['rating'][
                    'size'] >= 150  # here the popularMovies is like a condition that is going to be used for the moviestats inorder to imply and get 100+ above rated movies
pop15movies = movieStats[popularMovies].sort_values(by=[('rating', 'mean')], ascending=False)[:15]
# print(pop15movies)

df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
# print(df)
dfTop15 = df.sort_values(by=['similarity'], ascending=False)[:15]
print(dfTop15)
