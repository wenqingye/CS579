# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    
    >>> movies = pd.DataFrame([[123, 'Horror|Romance|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance', 'romance'], ['sci-fi']]
    """
    ###TODO

    # add a new column "tokens"
    movies['tokens'] = ''
    
    # extract from genres, list of strings
    for i in movies.index:
        movies.at[i, 'tokens'] = tokenize_string(movies.at[i, 'genres'])

    return movies
        

def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO

    # add new column features
    movies['features'] = ''

    # get all terms
    all_genres = []
    for token in movies['tokens'].tolist():
        all_genres.extend(token)
    all_genres = sorted(set(all_genres))

    # vocab, dict from term to int
    vocab = {}
    for i in range(len(all_genres)):
        vocab[all_genres[i]] = i

    # number of movies
    N = len(movies.index)

    # number of unique documents containing term i
    num_doc_i = {}
    for term in all_genres:
        num_doc_i[term] = 0
    for term in movies['tokens'].tolist():
        for t in set(term):
            num_doc_i[t] += 1

    # for each movie, calculate tfidf for each term and add to matrix
    for i in range(len(movies.index)):
        cur_terms = movies.at[i, 'tokens']
        cur_term_freq = {}
        for term in cur_terms:
            if term in cur_term_freq:
                cur_term_freq[term] += 1
            elif term not in cur_term_freq:
                cur_term_freq[term] = 1
        max_freq = sorted(cur_term_freq.items(), key=lambda x: -x[1])[0][1]
        #matrix
        row = []
        col = []
        data = []
        for term in cur_terms:
            tfidf = cur_term_freq[term] / max_freq * math.log10(N/num_doc_i[term])
            row.append(0)
            col.append(vocab[term])
            data.append(tfidf)
        #build matrix
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        matrix = csr_matrix((data, (row, col)),shape=(1, len(all_genres)))
        movies.at[i, 'features'] = matrix

    return (movies, vocab)
            

def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO

    # a, b to array
    #if they have value in the same position
    arr_a = a.toarray()[0]
    arr_b = b.toarray()[0]
    lengh = len(a.toarray()[0])
    dot_a_b = 0
    norm_a = 0
    norm_b = 0
    
    for i in range(lengh):
        if arr_a[i] != 0 and arr_b[i] != 0:
            dot_a_b += arr_a[i] * arr_b[i]
        if arr_a[i] != 0:
            norm_a += arr_a[i] * arr_a[i]
        if arr_b[i] != 0:
            norm_b += arr_b[i] * arr_b[i]

    cos_sim = dot_a_b / (math.sqrt(norm_a) * math.sqrt(norm_b))

    return cos_sim


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO

    predicted_rating = []
    ratings_train.index = range(ratings_train.shape[0])
    ratings_test.index = range(ratings_test.shape[0])

    # movies to dict
    all_movies = []
    for i in range(len(movies.index)):
        d = {}
        d['movieId'] = movies.at[i, 'movieId']
        d['features'] = movies.at[i, 'features']
        all_movies.append(d)

    # ratings_train to dict
    train = []
    for i in range(len(ratings_train.index)):
        d = {}
        d['userId'] = ratings_train.at[i, 'userId']
        d['movieId'] = ratings_train.at[i, 'movieId']
        d['rating'] = ratings_train.at[i, 'rating']
        train.append(d)
        
    # for each movie i in test, calculate predicted rating
    for i in range(len(ratings_test.index)):
        user = ratings_test.at[i, 'userId']
        movie = ratings_test.at[i, 'movieId']
        rate = []
        w = []
        no_pos_rt = []
        a = np.matrix([])
        for m in all_movies:
            if m['movieId'] == movie:
                a = m['features']
        # find all moveis this user rated in train
        for x in train:
            if x['userId'] == user:
                mid = x['movieId']
                rt = x['rating']
                b = np.matrix([])
                for m in all_movies:
                    if m['movieId'] == mid:
                        b = m['features']
                weight = cosine_sim(a, b)
                if weight > 0:
                    rate.append(rt)
                    w.append(weight)
                else:
                    no_pos_rt.append(rt)
        # weighted average
        if len(rate) > 0:
            for r in range(len(rate)):
                rate[r] = rate[r] * w[r] / sum(w)
            predicted_rating.append(sum(rate))
        else:
            predicted_rating.append(np.asarray(no_pos_rt).mean())

    return np.asarray(predicted_rating)
                

def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
