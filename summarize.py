"""
Summarize data.
"""
import cluster
import classify
import collect
import numpy as np


def main():
    users = cluster.read_file(collect.TWITTER_USER_DATA)['results']
    entire_data = cluster.read_file(collect.ENTIRE_STATUS)

    clusters = cluster.create_graph_clusters()
    test_data, train_data = classify.split_data_helper(entire_data)
    clf, label_test, predictions, x= classify.get_predictions(test_data, train_data)

    with open('summary.txt', 'w') as outfile:
        outfile.write("Number of users collected : %d" % len(users))

        outfile.write("\n Number of messages collected: %d" % len(entire_data['results']))

        outfile.write("\n Number of communities discovered: %d" % len(clusters))
        outfile.write("\n Total number of users first community: %d , second community : %d " % (clusters[0].order(), clusters[1].order()))
        outfile.write("\n Number of instances for positive sentiments are %d , negative sentiments are %d , neutral sentiments are %d " %
          (len(np.where( predictions == 1)[0]), len(np.where( predictions == -1)[0]),len(np.where( predictions == 0)[0])))
        outfile.write(("\n One example from each class: Positive Sentiment: %d , negative sentiment: %d, neutral sentiment: %d " % (1,-1,0)))



if __name__ == "__main__":
    main()
