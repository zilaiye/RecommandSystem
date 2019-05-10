#-*-coding:utf8-*-
"""
    author: kylin 
    date: 2019-05-08 
    this is a content based recommand system 
"""
import numpy as np 
import pandas as pd 
import os 
from operator  import itemgetter 

def get_avg_score(input_file,sep=','):
    """
        Args:
            input_file -> the movie rating file path
            sep -> column seperator 
        Output:
            a dict with key is movieId , value is the avg score 
    """
    rating_df = pd.read_csv(input_file,sep=sep)
    movieIdStat = rating_df.groupby(['movieId'],as_index=False)['userId','rating'].agg({'userId':'count','rating':'sum'}).rename(columns = {'userId':'user_nums','rating':'total_rating'})
    movieIdStat['avg_score'] =  round(movieIdStat['total_rating']*1.0/movieIdStat['user_nums'],3)
    movieIdAvgScore = movieIdStat[['movieId','avg_score']]
    movieIdAvgScore['movie_avg_score'] = movieIdAvgScore.apply(lambda r: {r['movieId']:r['avg_score']}, axis = 1)
    movie_avg_score = {} 
    _ = movieIdAvgScore['movie_avg_score'].apply(lambda  d: movie_avg_score.update(d))
    return movie_avg_score

def get_movie_category(input_file):
    """
        Args:
            input_file -> the movie and category file 
        Output:
            a dict , key is the movieId , values is the genres and  weight list  
    """
    movie_df = pd.read_csv(input_file)
    movie_df['genres_list'] = movie_df['genres'].apply(lambda x: x.split("|"))
    movie_df['genres_list_len'] = movie_df['genres_list'].apply(lambda x: len(set(x)))
    movie_df['item_ratio'] = movie_df['genres_list'].apply(lambda l : { e: round(1.0/len(l),3) for e in l } )
    movie_df['item_genres_ratio'] = movie_df.apply(lambda x: {x['movieId'] : x['item_ratio']} ,axis = 1)
    movieId_genres_ratio_map = {}
    _ = movie_df['item_genres_ratio'].apply(lambda r: movieId_genres_ratio_map.update(r))
    # 获得每个类下面，每个movie的倒排列表
    return movieId_genres_ratio_map

def get_category_movieId_reverse_map(movieId_genres_ratio_map,avg_score):
    """
        Args: 
            movieId_genres_ratio_map ->  a dict , key is the movieId , and value is a list of genres and weight just like (genres, weight) 
            avg_score -> each movieId's avg score 
        Output:
            a dict , key is the genres , values is a list of (movieId, avgscore) , in reserve order 
    """
    record = {}
    for itemId , genres_list in movieId_genres_ratio_map.items():
        if itemId not in avg_score:
            continue
        for genres in genres_list:
            if genres in record :
                record[genres].append((itemId,avg_score.get(itemId)))
            else:
                record[genres] = [ (itemId,avg_score.get(itemId)) ]
    # 按照倒排来一把
    for genres , title_score_list in record.items() :
        record[genres] = sorted(title_score_list, key = itemgetter(1), reverse=True)
    return record 


def get_time_score(timestamp):
    """
        Args: 
            timestamp -> the rating timestamp 
        Output:
            the score decay by month , the fresher the higher 
    """
    fix_time_stamp = 1476640644 + 1 
    seconds = fix_time_stamp - timestamp 
    seconds_per_day =   24*60*60  
    months = seconds*1.0/(30*seconds_per_day)
    score = round(1.0/(1+months),3)
    return score

def get_genres_score(r):
    """
        Args:
            r -> one row of a dataframe 
        Output:
            a dict that indicate the user's preference of a certain genres 
    """
    result = {}
    timescore = get_time_score(r['timestamp'])
    for genres,share in r['genres'].items():
        score = timescore*r['rating']*share
        if genres not in result:
            result[genres] = score
        else:
            result[genres] += score
    return {r['userId']:result} 

def get_user_profile(movieId_genres_ratio_map,rating_file,topK=2):
    """
        a function used to get the user profile 
        Args:
            movieId_genres_ratio_map ->  a dict , indicate the movies ratios in each genres 
            rating_file -> the rating file path 
            topK -> want top N ? 
        Output:
            a dict , that represent the user's preference to each genres , the preference is normalized ,
            key is the user , value is a list of tuple( genres , love_degree )
    """
    ratings = pd.read_csv(rating_file)
    ratings_filter = ratings[ (ratings['rating'] > 4.0 ) & ratings['movieId'].isin(movieId_genres_ratio_map.keys())]
    ratings_filter['genres'] = ratings_filter['movieId'].apply(lambda id: movieId_genres_ratio_map.get(id) )
    ratings_filter['genres_score'] = ratings_filter.apply( lambda x : get_genres_score(x), axis=1 )
    user_movie_score = {}
    _ = ratings_filter['genres_score'].apply(lambda d : user_movie_score.update(d))
    user_profile = {} 
    for userId in user_movie_score:
        if userId not in user_profile :
            user_profile[userId] = [] 
        total_score = 0 
        for zuhe in sorted(user_movie_score[userId].items(),key=itemgetter(1),reverse=True)[:topK]:
            user_profile[userId].append((zuhe[0],zuhe[1]))
            total_score += zuhe[1]
        for index in range(len(user_profile[userId])):
            lst = list(user_profile[userId][index])
            lst[1] = round(user_profile[userId][index][1]/total_score,3)
            user_profile[userId][index] = tuple(lst)
    return user_profile 

def recommand(userId,genres_movie_dict,user_profile,topK=10):
    """
        a function to recommand to a user based on the user profile 
        Args: 
            userId -> the user id 
            genres_movie_dict -> a dict , key is genres , value is a list of (movieId , score) , in reverse order 
            user_profile -> a dict , key is the userId , value is the user's preference 
            topK -> want how much of the recommand movies 
        Output:
            result with movieId and avg score 
    """
    result = []
    if userId not in user_profile:
        return result 
    top_genres = user_profile[userId]
    for genres , ratio in top_genres:
        num = int(ratio*topK) + 1 
        result +=  genres_movie_dict[genres][:num] 
    return result 


def run():
    path = os.path.abspath(os.getcwd())
    rating_file = os.path.join(path,"data/ml-latest-small/ratings.csv")
    print(rating_file)
    movie_file = os.path.join(path,"data/ml-latest-small/movies.csv")
    avg_score = get_avg_score(input_file=rating_file)
    itemId_genres_ratio_map = get_movie_category(movie_file)
    record = get_category_movieId_reverse_map(itemId_genres_ratio_map,avg_score)
    user_profile = get_user_profile(itemId_genres_ratio_map,rating_file)
    result = recommand(653,record,user_profile)
    print(result) 


if __name__ == '__main__':
    run()

