# coding: utf-8

"""
CS579: Assignment 0
Collecting a political social network

In this assignment, I've given you a list of Twitter accounts of 4
U.S. presedential candidates.

The goal is to use the Twitter API to construct a social network of these
accounts. We will then use the [networkx](http://networkx.github.io/) library
to plot these links, as well as print some statistics of the resulting graph.

1. Create an account on [twitter.com](http://twitter.com).
2. Generate authentication tokens by following the instructions [here](https://dev.twitter.com/docs/auth/tokens-devtwittercom).
3. Add your tokens to the key/token variables below. (API Key == Consumer Key)
4. Be sure you've installed the Python modules
[networkx](http://networkx.github.io/) and
[TwitterAPI](https://github.com/geduldig/TwitterAPI). Assuming you've already
installed [pip](http://pip.readthedocs.org/en/latest/installing.html), you can
do this with `pip install networkx TwitterAPI`.

OK, now you're ready to start collecting some data!

I've provided a partial implementation below. Your job is to complete the
code where indicated.  You need to modify the 10 methods indicated by
#TODO.

Your output should match the sample provided in Log.txt.
"""

# Imports you'll need.
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import configparser
from TwitterAPI import TwitterAPI

%matplotlib inline

consumer_key = 'MT3pZKCx4Zzs1uKUsKva5sRBk'
consumer_secret = 'P7jFIn5tu8wwwywvSsyfgFUjkiFffk2KfCheaCbELwTfGNOfz3'
access_token = '2296782132-uvsOo6QXexHihcpkYIK4PfNDmhPQePSL2dLdT7k'
access_token_secret = 'Fk5aqNzaUb6I2f9asKBssYcQdk8Y5CoEZAtJJEqrWNygC'







# This method is done for you. 
def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    #already done, return TwitterAPI(), don't need to get again
    """
    config = configparser.Configparser()
    config.read(config_file)
    twitter = TwitterAPI(
        config.get('twitter', 'consumer_key'),
        config.get('twitter', 'consumer_secret'),
        config.get('twitter', 'access_token'),
        config.get('twitter', 'access_token_secret'))
    #return twitter
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
    


def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.

    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.

    Here's a doctest to confirm your implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    ###TODO
    #pass

    with open(filename, 'r') as f:
        #content = f.readlines()
        content = f.read().splitlines()

    return content
    #content is the screen_names needed, list


# I've provided the method below to handle Twitter's rate limiting.
# You should call this method whenever you need to access the Twitter API.
def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)

    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup

    In this example, I test retrieving two users: twitterapi and twitter.

    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    ###TODO
    #pass

    users = []
    for sname in screen_names:
        request = robust_request(twitter, 'users/lookup', {'screen_name': sname}, max_tries=5)
        user = [i for i in request]
        b = {'screen_name': user[0]['screen_name'],
             'id': user[0]['id'],
             'location': user[0]['location']}
        users.append(b)
   
    return users


    


def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids

    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.

    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.

    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.

    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    #pass

    friends = []
    request = robust_request(twitter, 'friends/ids', {'screen_name': screen_name, 'count': 5000}, max_tries=5)
    friends = sorted([i for i in request])
    
    #print(len(friends))
    #print(friends)

    return friends



def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.

    Store the result in each user's dict using a new key called 'friends'.

    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing

    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    ###TODO
    #pass
    
    for user in users:
        user['friends'] = get_friends(twitter, user['screen_name'])
        #get list of friends ids
        #print(len(user['friends']))

def print_num_friends(users):
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    ###TODO
    #pass
    
    for user in users:
        num_friends = str(len(user['friends']))
        #print(num_friends)
        print (user['screen_name'] + ' ' + num_friends)




def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter

    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    ###TODO
    #pass
    
    cnt = Counter()
    friends = []
    for user in users:
        friends.append(user['friends'])
        #list of friends of each user
    for friend in friends:
        cnt.update(friend)
        #count each friend in friends[] and update the counter
    return cnt



def friend_overlap(users):
    """
    Compute the number of shared accounts followed by each pair of users.

    Args:
        users...The list of user dicts.

    Return: A list of tuples containing (user1, user2, N), where N is the
        number of accounts that both user1 and user2 follow.  This list should
        be sorted in descending order of N. Ties are broken first by user1's
        screen_name, then by user2's screen_name (sorted in ascending
        alphabetical order). See Python's builtin sorted method.

    In this example, users 'a' and 'c' follow the same 3 accounts:
    >>> friend_overlap([
    ...     {'screen_name': 'a', 'friends': ['1', '2', '3']},
    ...     {'screen_name': 'b', 'friends': ['2', '3', '4']},
    ...     {'screen_name': 'c', 'friends': ['1', '2', '3']},
    ...     ])
    [('a', 'c', 3), ('a', 'b', 2), ('b', 'c', 2)]
    """
    ###TODO
    #pass

    count = 0
    friend_overlap = []
    overlap = tuple()
    
    for i in range(0, len(users)):
        for j in range(i+1, len(users)):
            for m in range(0, len(users[i]['friends'])):
                for n in range(0, len(users[j]['friends'])):
                    if users[i]['friends'][m] == users[j]['friends'][n]:
                        count += 1
            overlap = (users[i]['screen_name'], users[j]['screen_name'], count)
##            overlap[0] = str(users[i]['screen_name'])
##            overlap[1] = str(users[j]['screen_name'])
##            overlap[2] = count
            friend_overlap.append(overlap)
            count = 0

    friend_overlap = sorted(friend_overlap, key=lambda tup: (-tup[2], tup[0], tup[1]))

    return friend_overlap                       




def followed_by_hillary_and_donald(users, twitter):
    """
    Find and return the screen_name of the one Twitter user followed by both Hillary
    Clinton and Donald Trump. You will need to use the TwitterAPI to convert
    the Twitter ID to a screen_name. See:
    https://dev.twitter.com/rest/reference/get/users/lookup

    Params:
        users.....The list of user dicts
        twitter...The Twitter API object
    Returns:
        A string containing the single Twitter screen_name of the user
        that is followed by both Hillary Clinton and Donald Trump.
    """
    ###TODO
    #pass

    fid = 0

    hillary = [x for x in users if x['screen_name'] == 'HillaryClinton']
    #return hillary
    donald = [x for x in users if x['screen_name'] == 'realDonaldTrump']
    #return len(donald[0]['friends'])
    

    #hillary = filter(lambda user: user['screen_name'] == 'HillaryClinton', users)
    #donald = filter(lambda user: user['screen_name'] == 'realDonaldTrump', users)
    for i in range(0, len(hillary[0]['friends'])):
        for j in range(0, len(donald[0]['friends'])):
            if hillary[0]['friends'][i] == donald[0]['friends'][j]:
                fid = hillary[0]['friends'][i]

    #print(type(fid))
    #return fid
    request = robust_request(twitter, 'users/lookup', {'user_id': fid}, max_tries=5)
    r = [i for i in request]
    #return r[0]
    fname = r[0]['screen_name']

    return fname



def create_graph(users, friend_counts):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)

        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.

    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    ###TODO
    #pass
    
    #friend_counts = count_friends(users)
    friends = [x for x in friend_counts if friend_counts[x] > 1]
    #list of friends ids followed by more than one user

    #print(friends)

    #draw undirected graph
    graph = nx.Graph()
    #add nodes for friends
    for x in friends:
        graph.add_node(x)
    #add users nodes
    for user in users:
        graph.add_node(user['screen_name'])
        #list of friends should be plotted
        fndlst = set(user['friends']) & set(friends)
        #print(fndlst)
        #add edges for each node
        for fnd in fndlst:
            #graph.add_node(fnd)
            graph.add_edge(fnd, user['screen_name'])

    nx.draw(graph, with_labels=True)
    return graph
    



def draw_network(graph, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).

    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.

    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    ###TODO
    #pass

    #only users have lables
    label = {n: n if n in (u['screen_name'] for u in users) else '' for n in graph.nodes()}
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    #pos = nx.spring_layout(graph, k=.04, dim=2, scale=2)

    nx.draw_networkx(graph, labels=label, alpha=.5, node_size=100, width=.1)
    plt.savefig(filename)
    



def main():
    """ Main method. You should not modify this. """
    #print("import done")
    twitter = get_twitter()
    #print("get twitter done")
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)

    #users = get_users(twitter, ['twitterapi', 'twitter'])
    #print(users)

    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
 
    #print(get_friends(twitter, 'aronwc')[:5])

    #users = [{'screen_name': 'realDonaldTrump'}]
    add_all_friends(twitter, users)
    #print(users)
    
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))

    #print(users[3]['friends'][41])
    
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))

    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
    main()

# That's it for now! This should give you an introduction to some of the data we'll study in this course.
