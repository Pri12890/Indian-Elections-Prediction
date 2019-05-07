import cluster
from collect import ENTIRE_STATUS
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt

def get_tweets(data):
    """
     Here, we will fetch the tweets for classification
    :param data: - This is the complete twitter search response data taken from json file
    :return: - tweets - These are the tweets fetched from user's data
    """
    tweets = []
    list_results = data['results']
    for x in list_results:
        text_ = x['text']
        tweets.append(text_)
    print("tweets collected: ", len(tweets))
    return tweets


def main():
    # data = cluster.read_file(ENTIRE_STATUS)
    # tweets = get_tweets(data)
    # vectorizer = CountVectorizer(min_df=2, ngram_range=(1,1))
    # X= vectorizer.fit_transform(tweets)
    # print('vectorized %d tweets, found %d terms' % (X.shape[0], X.shape[1]))
    # print(vectorizer.vocabulary_)
    # print("abc")

    N = 50
    np.random.seed(19680801)
    colors = np.random.rand(N)
    area = (30 * np.random.rand(N)) ** 2
    predictions=[1,2,3,4,5,6,1,2,3,4,5,6, 8,4,5,6]
    label_test = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 6,3,5,1]
    plt.scatter(predictions, label_test,s=area, c= colors,alpha=0.5)
    plt.show()
    plt.savefig("scatter1.png")


if __name__ == '__main__':
    main()