import pandas as pd
import numpy as np
desired_width=320
#This just for viewing the dataframe in the o/p
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',12)




r_cols=['user_id','movie_id','rating']
ratings=pd.read_csv("C:\\Users\\Shreyas\\MLCourse\\ml-100k\\u.data",sep="\t",names=r_cols,usecols=range(3))

m_cols=['movie_id','title']
movieRating=pd.read_csv("C:\\Users\\Shreyas\\MLCourse\\ml-100k\\u.item",sep="|",names=m_cols,usecols=range(2),encoding='latin-1')

ratings=pd.merge(movieRating,ratings)
# print(ratings)

userRatings=ratings.pivot_table(index=['user_id'],values='rating',columns=['title'])
# print(userRatings)

corrMatrix=userRatings.corr()
# print(corrMatrix)

corrMatrix=userRatings.corr(method='pearson',min_periods=150)#min_period is like the rating number o.e how many have rated that movie method is the algo to be used while corr is taking place
# print(corrMatrix['Empire Strikes Back, The (1980)'])

myRatings=userRatings.loc[0].dropna()
print(myRatings)
simCandidate=pd.Series(dtype='float64')
for i in range(0,len(myRatings.index)):
    print("Adding similar movies for "+myRatings.index[i]+"....")
    #here we are going to retrive similar movies
    sims=corrMatrix[myRatings.index[i]].dropna()
    #Now we are going to scale its similarity by how well we have rated the movie
    sims=sims.map(lambda x:x*myRatings[i])
    # print(sims)
    #adding sims to simcandidate
    simCandidate=simCandidate.append(sims)
print('Sorting...')
simCandidate.sort_values(inplace=True,ascending=False)
# print(simCandidate[:25])
simCandidate=simCandidate.groupby(simCandidate.index).sum()
simCandidate=simCandidate.drop(myRatings.index)
simCandidate.sort_values(ascending=False,inplace=True)
print(simCandidate[:20])


