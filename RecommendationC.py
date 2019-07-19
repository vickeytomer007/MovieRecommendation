import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csc_matrix
import sklearn.preprocessing as pp
from sklearn.metrics.pairwise import cosine_similarity

def Reco(df_movies):
    global similarity
    unique_genre=[]
    for inx,row in df_movies.iterrows():
        genre=row['genres'].split('|')
        for gen in genre:
            if gen not in unique_genre:
                unique_genre.append(gen)
    x_len=df_movies['movieId'].count()
    y_len=len(unique_genre)
    x=np.zeros((x_len,y_len))    
    XDF=pd.DataFrame(x,columns=unique_genre,index=df_movies.movieId)
    for ix,rowdata in df_movies.iterrows():
        genre=rowdata['genres'].split('|')
        for gen in genre:
          XDF.loc[rowdata['movieId'],gen]=1
    from sklearn.metrics.pairwise import cosine_similarity
    similarity=cosine_similarity(XDF,XDF)
    similarity=pd.DataFrame(similarity,columns=df_movies.title,index=df_movies.title)
    #return similarity
def simi(titles,x=5):
    print(similarity[titles].sort_values(ascending=False).head(x))

def cosine_similarities(mat):
        col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
        return col_normed_mat.T * col_normed_mat
def cosine_similarity_Movie(dt): 
    global cosine_movie
    global dataframe
    dataframe=dt
    #dataframe.drop(['genres','timestamp'],axis=1,inplace=True)
    dataframe['MID']=dataframe['movieId'].astype('category').cat.codes   
    dataframe['UID']=dataframe['userId'].astype('category').cat.codes 
    mat =csc_matrix((dataframe["rating"],(dataframe["MID"],dataframe["UID"])))
    cosine_movie=cosine_similarity(mat, Y=None, dense_output=False)
    
def MovieToMovie(title):
    #title='Toy Story (1995)'
    id=(dataframe[dataframe['title']==title]['MID'].head().drop_duplicates())
    RM_id=np.argsort(cosine_movie.getrow(id).toarray()).T[-6:-1]
    rm=dataframe.loc[dataframe['MID'].isin(RM_id)][['title']].drop_duplicates() #.sort_values(by=RM_id)
    return rm


