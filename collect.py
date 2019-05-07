"""
Collect data.
"""
import sys
import time
from collections import defaultdict
from TwitterAPI import TwitterAPI
from textblob import TextBlob
import regex as re
import json

'''
In this assignment, we will try to make predictions about the Indian Elections 2019, using the tweets posted recently
about the elections going on in India.

collect.py collects all tweets with matching query into collected_tweets.json
'''

CONSUMER_KEY = 'iEdoYDoRqaDuk5E5a3DMq76lL'
CONSUMER_SECRET = 'IWrTAVKtOcVTsFHDNKY9ZUQyR8iqy3ysTG16vBemCFAxPzWFIi'
ACCESS_TOKEN = '1086781608135483392-yReuRuxaEv4fAsaEjKzoniGQy3WTJW'
ACCESS_TOKEN_SECRET = 'uF5gMBQt61VHbn9j7187gMhhgbNBnxzZS6iBV7HfwbYOv'

SEARCH_CONGRESS = ['congress', 'gandhi', 'sonia']
SEARCH_BJP = ['modi', 'bjp', 'narendramodi']
ENTIRE_STATUS = 'collected_tweets.json'

BJP_POS_USER_FILE = 'bjp_pos_user.json'
CON_POS_USER_FILE = 'con_pos_user.json'
BJP_NEG_USER_FILE = 'bjp_neg_user.json'
CON_NEG_USER_FILE = 'con_neg_user.json'
FRIENDS_FOR_NEUTRAL_PEOPLE = 'friends_for_neutral_people.json'
FRIENDS_EDGES = 'friends_edges.json'
TWITTER_USER_DATA = 'twitter_user_data.json'

SEARCH_FOR = 'modi OR rahul AND gandhi OR sonia OR IndiaElection2019 OR Poll2019India OR narendramodi OR congressvsbjp OR rahulvsmodi OR bjp OR congress AND India'
MAX_TWEETS = 5000
MAX_NEUTRAL_PEOPLE = 10

BJP_POS_USER = set()
BJP_NEG_USER = set()
CONG_POS_USER = set()
CONG_NEG_USER = set()


def get_twitter():
    """ construct an instance of TwitterAPI.
    Returns:
      An instance of TwitterAPI.
    """

    api = TwitterAPI(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    print("Established Twitter Connection")
    return api


def get_election_tweets(twitter):
    """
    :param twitter:  An instance of TwitterAPI.
    :return: list of twitter query search responses called status
    """
    params = 'q=' + SEARCH_FOR
    params += '&count=200&include_entities=true'
    entire_status = []
    while True or len(entire_status) < MAX_TWEETS:
        try:
            response = twitter.request('search/tweets', params=params)
            json_response = response.json()
            params = json_response['search_metadata']['next_results'].strip('?')
            found_tweets = json_response['statuses']
            entire_status = entire_status + found_tweets
        except KeyError as e:
            print ('Found error', e, 'during get_election_tweets')
            if json_response['errors'][0]['code'] == 88 and len(entire_status) > MAX_TWEETS:
                print ('Hitting rate limiting, stopping tweet collection')
                # time.sleep(15 * 60)
                break
        except:
            e = sys.exc_info()[0]
            print ('Found error', e, 'during get_election_tweets')
    print ('Writing %d twitter search responses to %s' % (len(entire_status), ENTIRE_STATUS))
    persist_data(entire_status, ENTIRE_STATUS)
    return entire_status


def clean_collected_data(entire_status):
    """
    Helper method to clean and process twitter data. Groups tweets by user id
    :param entire_status: list of twitter search responses called status
    :return: set of neutral people, who have positive views about both party
    """
    user_and_tweets = defaultdict(list)
    for status in entire_status:
        tweet = status['text']
        user_id = status['user']['id']
        user_and_tweets[user_id].append(tweet)
        category_of_user(tweet, user_id)
    print ('Grouped %d tweets across %d users, writing grouped data to %s' % (len(entire_status), len(user_and_tweets),
                                                                              TWITTER_USER_DATA))
    persist_data(user_and_tweets, TWITTER_USER_DATA)
    common_pos_people = BJP_POS_USER.intersection(CONG_POS_USER)
    return common_pos_people


def get_friends(twitter, user_id, should_sleep=False):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids


    Args:
        twitter.......The TwitterAPI object
        user_id... a string of a Twitter screen name
        should_sleep... should sleep on rate limiting error
    Returns:
        A list of ints, one per friend ID.

    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.

    In this test case, I return the first 5 accounts that I follow.
    """
    try:
        response = twitter.request('friends/ids', params='user_id=' + str(user_id) + '&count=5000')
        json_response = response.json()
        user_information = json_response['ids']
        return set(user_information)
    except KeyError as e:
        print ('Found error ', e, 'during get friends')
        if json_response['errors'][0]['code'] == 88 and should_sleep:
            print('Hitting rate limiting, sleeping for 15 mins')
            time.sleep(15 * 60)
            return get_friends(twitter, user_id, False)
        else:
            print('Hitting rate limiting during get friends, skip sleeping ')
            return set()


def category_of_user(tweet, user_id):
    """
    labels user in one of the four category based on his tweets.
    BJP_POS_USER, BJP_NEG_USER, CONG_POS_USER, CONG_NEG_USER
    :param tweet: tweet for performing sentiment analysis
    :param user_id: user to categorize
    :return: nothing
    """
    result_cong = any([re.search(w, tweet.lower()) for w in SEARCH_CONGRESS])
    result_bjp = any([re.search(w, tweet.lower()) for w in SEARCH_BJP])

    sentiment = get_sentiment(tweet)
    if result_bjp and not result_cong:
        if sentiment == 'positive':
            BJP_POS_USER.add(user_id)
        elif sentiment == 'negative':
            BJP_NEG_USER.add(user_id)

    if result_cong and not result_bjp:
        if sentiment == 'positive':
            CONG_POS_USER.add(user_id)
        elif sentiment == 'negative':
            CONG_NEG_USER.add(user_id)


def persist_data(data_to_persist, file_name):
    """
    Writes data to file
    :param data_to_persist: data to be written
    :param file_name: file to be written into
    :return: nothing
    """
    data = {'results': data_to_persist}
    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)


def get_sentiment(tweet):
    """
    Helper function to label tweet in positive, neutral, negative sentiment
    :param tweet: tweet to be analysed
    :return: sentiment as str
    """
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


def find_friends(twitter, neutral_people):
    """
    Find friends for list of neutral_people, and find friends of friends with same views
    :param twitter: twitter instance
    :param neutral_people: list of neutral people
    :return: nothing
    """
    all_bjp_pos_friends = set()
    all_cong_pos_friends = set()
    all_bjp_neg_friends = set()
    all_cong_neg_friends = set()
    friends_edges = dict()
    friends_for_neutral_people = defaultdict(dict)
    count = 0
    for p in neutral_people:
        if count > MAX_NEUTRAL_PEOPLE:
            break
        count += 1
        friends = get_friends(twitter, p, True)
        bjp_pos_friends = friends.intersection(BJP_POS_USER)
        bjp_neg_friends = friends.intersection(BJP_NEG_USER)
        cong_pos_friends = friends.intersection(CONG_POS_USER)
        cong_neg_friends = friends.intersection(CONG_NEG_USER)
        friends_for_neutral_people[p]['bjp_pos'] = list(bjp_pos_friends)
        friends_for_neutral_people[p]['bjp_neg'] = list(bjp_neg_friends)
        friends_for_neutral_people[p]['con_pos'] = list(cong_pos_friends)
        friends_for_neutral_people[p]['con_neg'] = list(cong_neg_friends)
        all_bjp_pos_friends = all_bjp_pos_friends.union(bjp_pos_friends)
        all_cong_pos_friends = all_cong_pos_friends.union(cong_pos_friends)
        all_bjp_neg_friends = all_bjp_neg_friends.union(bjp_neg_friends)
        all_cong_neg_friends = all_cong_neg_friends.union(cong_neg_friends)

    real_bjp_pro_friends = all_bjp_pos_friends.intersection(all_cong_neg_friends)
    real_con_pro_friends = all_cong_pos_friends.intersection(all_bjp_neg_friends)
    print ('Found %d strict BJP supporter friends for %d neutral people, best effort finding friends of friends' % (
        len(real_bjp_pro_friends), len(neutral_people)))
    print (
            'Found %d strict Congress supporter friends for %d neutral people, best effort finding friends of friends' % (
        len(real_bjp_pro_friends), len(neutral_people)))
    find_friends_edges(all_bjp_neg_friends, all_bjp_pos_friends, all_cong_neg_friends, all_cong_pos_friends,
                       friends_edges, real_bjp_pro_friends, twitter)
    find_friends_edges(all_bjp_neg_friends, all_bjp_pos_friends, all_cong_neg_friends, all_cong_pos_friends,
                       friends_edges, real_con_pro_friends, twitter)

    print ('Writing %d BJP Positive friends, '
           '%d BJP Negative friends,'
           ' %d Congress Positive friends, '
           '%d Congress Negative friends for'
           ' %d neutral people to %s' % (
        len(all_bjp_pos_friends),
        len(all_bjp_neg_friends),
        len(all_cong_pos_friends),
        len(all_bjp_neg_friends),
        len(neutral_people),
        FRIENDS_FOR_NEUTRAL_PEOPLE))
    persist_data(friends_for_neutral_people, FRIENDS_FOR_NEUTRAL_PEOPLE)
    print ('Writing friends of neutral people\'s %d friends to %s ' % (len(friends_edges), FRIENDS_EDGES))
    persist_data(friends_edges, FRIENDS_EDGES)


def find_friends_edges(all_bjp_neg_friends, all_bjp_pos_friends, all_cong_neg_friends, all_cong_pos_friends,
                       friends_edges, real_party_friends, twitter):
    """
    Helper function to find friends of real_party_friends, and store friends who have any views about any party
    :param all_bjp_neg_friends: All known bjp neg friends
    :param all_bjp_pos_friends: All known bjp pos friends
    :param all_cong_neg_friends: All known congress neg friends
    :param all_cong_pos_friends: All known congress pos friends
    :param friends_edges: data structure to store friend ids - who belong to any of above mentioned list
    :param real_party_friends: set of people to find friends for
    :param twitter: instance of twitter
    :return: nothing
    """
    for p in real_party_friends:
        friends = get_friends(twitter, p)
        pos_bjp = friends.intersection(all_bjp_pos_friends)
        pos_con = friends.intersection(all_cong_pos_friends)
        neg_bjp = friends.intersection(all_bjp_neg_friends)
        neg_con = friends.intersection(all_cong_neg_friends)
        all_friends = list(pos_bjp.union(pos_con).union(neg_bjp).union(neg_con))
        friends_edges[p] = all_friends
        print ('Found %d friends with political views for neutral person\'s friend %s' % (len(all_friends), str(p)))


def main():
    print ('##################### Welcome to Indian Election 2019 Twitter Sentiment Analysis ########################')
    twitter = get_twitter()
    print('Collecting at least %d tweets related to Indian Prime Minister Election 2019 ' % MAX_TWEETS)
    entire_data = get_election_tweets(twitter)
    print('Cleaning and processing %d tweet search statuses' % len(entire_data))
    neutral_people = clean_collected_data(entire_data)
    print('Found %d neutral views people, finding friends for all' % len(neutral_people))
    find_friends(twitter, neutral_people)
    print ('Writing %d total BJP Positive views users in %s' %(len(BJP_POS_USER), BJP_POS_USER_FILE))
    persist_data(list(BJP_POS_USER), BJP_POS_USER_FILE)
    print ('Writing %d total BJP Negative views users in %s' % (len(BJP_NEG_USER), BJP_NEG_USER_FILE))
    persist_data(list(BJP_NEG_USER), BJP_NEG_USER_FILE)
    print ('Writing %d total Congress Positive views users in %s' % (len(CONG_POS_USER), CON_POS_USER_FILE))
    persist_data(list(CONG_POS_USER), CON_POS_USER_FILE)
    print ('Writing %d total Congress Negative views users in %s' % (len(CONG_NEG_USER), CON_NEG_USER_FILE))
    persist_data(list(CONG_NEG_USER), CON_NEG_USER_FILE)


if __name__ == '__main__':
    main()
