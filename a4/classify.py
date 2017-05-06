"""
classify.py

Classify collected tweets from original users by sentiment.
"""


import numpy as np
import pickle
import re


def read_afinn():
    """
    Read afinn document to dict, word to score.
    """
    
    afinn = dict()
    with open('sentiment.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                afinn[parts[0]] = int(parts[1])

    print('read %d AFINN terms.\nE.g.: %s' % (len(afinn), str(list(afinn.items())[:10])))

    return afinn


def get_tweets(name):
    """
    Load stored tweets.
    List of strings, one per tweet.
    """

    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.
    """

    #convert to lowercase
    doc = doc.lower()
    
    #convert to string with only alphanumerics and split
    if keep_internal_punct == False:
        token = re.sub('\W+', ' ', doc).split() 
    if keep_internal_punct == True:
        token = [re.sub('^\W+|\W+$', '', x) for x in doc.split()]

    #return numpy array
    return np.array(token)


pos_tweets = []
neg_tweets = []
neutral_tweets = []


def tweet_sentiment(tweet, afinn):
    """
    score of the tweet.

    paras:
        tweet........a tweet, string
        afinn........dict from word to score
    returns:
        nothing
    """
    
    score = 0

    #tokenize tweet
    terms = tokenize(tweet)
    
    for t in terms:
        if t in afinn:
            score += afinn[t]

    if score > 0:
        pos_tweets.append(tweet)
    if score < 0:
        neg_tweets.append(tweet)
    if score == 0:
        neutral_tweets.append(tweet)


def all_tweets_sentiment(tweets, afinn):
    """
    get sentiment for all tweets collected

    paras:
        tweets.......all tweets collected, list of strings
        afinn........dict from word to score
    returns:
        nothing
    """
    
    for tweet in tweets:
        tweet_sentiment(tweet, afinn)


def classify_results(pos_tweets, neg_tweets, neutral_tweets):
    """
    add all classes to a dict, each a list
    """

    classify_results = {}
    classify_results['pos'] = pos_tweets
    classify_results['neg'] = neg_tweets
    classify_results['neutral'] = neutral_tweets

    return classify_results


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
    

def main():
    afinn = read_afinn()
    print('afinn read.')
    tweets = get_tweets('tweets')
    print('tweets got.')
    all_tweets_sentiment(tweets, afinn)
    print('got sentiment for all tweets.')
    print('There are %d positive tweets, %d negative tweets and %d neutral tweets.' %
          (len(pos_tweets), len(neg_tweets), len(neutral_tweets)))
    cla_ret = classify_results(pos_tweets, neg_tweets, neutral_tweets)
    save_obj(cla_ret, 'classify_results')


if __name__ == '__main__':
    main()





    
