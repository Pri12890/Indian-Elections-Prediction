"""
In this assignment, we will try to make predictions about the Indian Elections 2019, using the tweets posted recently
about the elections going on in India.

collect.py collects all tweets with matching query into collected_tweets.json
"""
from nltk.corpus import stopwords
from textblob import TextBlob
import numpy as np
import cluster
import a2
from collect import ENTIRE_STATUS
from nltk.tokenize import word_tokenize
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


def train_test_split(all_tweets):
    """
    Here, we will split the data into training and test data
    :param all_tweets:
    :return: train_data, test Data - to perform classification
    """

    sampled_tweets = all_tweets
    train_data = sampled_tweets[:int(len(sampled_tweets) * 0.8)]
    test_data = sampled_tweets[int(len(sampled_tweets) * 0.8):]
    return train_data, test_data


def get_sentiment(tweet):
    """
    Here, we will classify a tweet as expressing positive (+1),
    negative (-1) or neutral(0) sentiment'
    :param tweet:
    :return: 1 for positive sentiment, -1 for negative sentiment, 0 for neutral sentiment
    We will compare it against the classifier's predictions and evaluate accuracy of the classifier
    """
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


def label_data(data):
    """
    In this function, we will label the tweets using the above get_sentiment
    function by calling it for all the tweets we fetched'
    :param data:
    :return: label_data_list - This contains labels(sentiments) of all the tweets
    """

    label_data_list = []
    for iter in data:
        sentiment = get_sentiment(iter)
        label_data_list.append(sentiment)
    label_data_list = np.array(label_data_list)
    return label_data_list


def classifier(trainData, testData):
    """
    We will fit a classifier in this function and evaluate the accuracy of the classifier using test data
    :param trainData:
    :param testData:
    :return:
    """
    clf, label_test, predictions, x = get_predictions(testData, trainData)

    print('testing accuracy after classification=',
          float(len(np.where(label_test == predictions)[0])) / float(len(label_test)))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    misclassified = a2.print_top_misclassified(testData, label_test, x, clf, 100)
    print('\n Total misclassified test documents are %d' % len(misclassified))

    false_prob = [i[2] for i in misclassified]
    true_prob = [i[4] for i in misclassified]

    plt.scatter(false_prob, true_prob)
    plt.xlabel('Predicted False Probability')
    plt.ylabel('Labeled True Probability')
    plt.title('Predicted Vs True Probability')
    plt.savefig("proba.png")
    plt.show()



    # scattter plot for predictions vs label_test
    plt.figure(2)
    N = 50
    np.random.seed(19680801)
    # colors = np.random.rand(N)
    area = (30 * np.random.rand(N)) ** 2
    plt.scatter(predictions, label_test,s=area, alpha=0.5)
    plt.title('Prediction Vs Truth Label')
    plt.savefig("scatter.png")
    plt.show()



def get_predictions(testData, trainData):
    label_train = label_data(trainData)
    label_test = label_data(testData)
    feature_fns = [a2.token_features, a2.token_pair_features, a2.lexicon_features]
    results = a2.eval_all_combinations(trainData, label_train, [True, False], feature_fns, [2, 5, 10])
    best_result = results[0]
    clf, vocab = a2.fit_best_classifier(trainData, label_train, results[0])
    tokens_list = [a2.tokenize(d, best_result.get("punct", False)) for d in testData]
    x, ignore = a2.vectorize(tokens_list, list(best_result['features']), best_result['min_freq'], vocab)
    predictions = clf.predict(x)
    return clf, label_test, predictions, x


def main():
    data = cluster.read_file(ENTIRE_STATUS)
    test_data, train_data = split_data_helper(data)
    classifier(train_data, test_data)


def split_data_helper(data):
    tweets = get_tweets(data)
    removed_stop_words_tweets = remove_stop_words(tweets)
    print("Splitting data into Training and Test data")
    train_data, test_data = train_test_split(removed_stop_words_tweets)
    return test_data, train_data


def remove_stop_words(tweets):
    print("removing stop words, to improve model's accuracy")
    filtered_tweets = []
    stop_words = set(stopwords.words('english'))
    for t in tweets:
        word_tokens = word_tokenize(t)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        filtered_tweets.append(" ".join(filtered_sentence))

    return filtered_tweets



if __name__ == '__main__':
    main()
