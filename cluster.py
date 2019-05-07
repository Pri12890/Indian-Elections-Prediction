from collections import defaultdict

import collect
import json
import networkx as nx
import matplotlib.pyplot as plt
import a1


def read_file(filename):
    """
    Simply reads the file
    :param filename:
    :return: data read from file
    """
    try:
        print ('reading file %s' % filename)
        with open(filename) as json_file:
            data = json.load(json_file)
            return data
    except:
        print ('File %s not found, returning empty data ' % filename)
        return defaultdict(list)


def create_graph():
    """
    Construct Graph from file written in collect.py.
    Files used to plot are FRIENDS_FOR_NEUTRAL_PEOPLE, FRIENDS_EDGES
    :return: graph
    """
    g = nx.Graph()
    plt.figure(figsize=(15, 25))
    plt.axis('off')
    json_data = read_file(collect.FRIENDS_FOR_NEUTRAL_PEOPLE)
    friends_data = json_data['results']
    pos_bjp = []
    pos_con = []
    neg_bjp = []
    neg_con = []
    neutral_list = []
    for f in friends_data.items():
        neutral_man = f[0]
        bjp_friends = f[1]['bjp_pos']
        con_friends = f[1]['con_pos']
        anti_bjp = f[1]['bjp_neg']
        anti_con = f[1]['con_neg']

        if not bjp_friends and not con_friends and not anti_bjp and not anti_con:
            continue
        else:
            neutral_list.append(neutral_man)
            g.add_node(neutral_man)
            pos_bjp = pos_bjp + bjp_friends
            pos_con = pos_con + con_friends
            neg_bjp = neg_bjp + anti_bjp
            neg_con = neg_con + anti_con
            all_friends_of_neutral = bjp_friends + con_friends + anti_bjp + anti_con
            g.add_nodes_from(all_friends_of_neutral)
            for ff in all_friends_of_neutral:
                g.add_edge(neutral_man, ff)

    json_data = read_file(collect.FRIENDS_EDGES)
    friends_data = json_data['results']
    for f in friends_data:
        f0 = f[0]
        f1 = f[1]
        for f in f1:
            g.add_edge(f0, f)
    draw_graph(g, pos_bjp, neg_bjp, neg_con, pos_con, neutral_list, 1,
               'graph.png', 'Graph containing all users of all communities - \n '
                            'Neutral Users - Purple | '
                            'Positive for BJP - Green | '
                            'Negative for BJP - Red | \n'
                            'Positive for Congress - Blue | '
                            'Negative for Congress - Yellow ')



    return g


def draw_clusters(clusters):
    """
    Draws the cluster after community detection following same color convention as actual graph
    :param clusters:
    :return: nothing
    """
    bjp_pos = read_file(collect.BJP_POS_USER_FILE)['results']
    set_bjp_pos = set(bjp_pos)
    bjp_neg = read_file(collect.BJP_NEG_USER_FILE)['results']
    set_bjp_neg = set(bjp_neg)
    con_pos = read_file(collect.CON_POS_USER_FILE)['results']
    set_con_pos = set(con_pos)
    con_neg = read_file(collect.CON_NEG_USER_FILE)['results']
    set_con_neg = set(con_neg)
    count = 2
    for cluster in clusters:
        cluster_bjp_pos = set()
        cluster_bjp_neg = set()
        cluster_con_pos = set()
        cluster_con_neg = set()
        cluster_neutral = set()
        for n in cluster.nodes():
            if n in set_bjp_pos:
                cluster_bjp_pos.add(n)
            elif n in set_bjp_neg:
                cluster_bjp_neg.add(n)
            elif n in set_con_pos:
                cluster_con_pos.add(n)
            elif n in set_con_neg:
                cluster_con_neg.add(n)
            else:
                cluster_neutral.add(n)
        draw_graph(cluster, cluster_bjp_neg, cluster_bjp_pos, cluster_con_neg, cluster_con_pos, cluster_neutral, count,
                   'cluster_' + str(count - 1), 'community detection - cluster '+ str(count - 1) + '\n Neutral Users - Purple | '
                                                                                                   'Positive for BJP - Green | '
                                                                                                   'Negative for BJP - Red | \n '
                                                                                                   'Positive for Congress - Blue | '
                                                                                                   'Negative for Congress - Yellow ')
        count += 1


def draw_graph(cluster, cluster_bjp_neg, cluster_bjp_pos, cluster_con_neg, cluster_con_pos, cluster_neutral, count,
               graph_name, title):
    """
    Helper method to draw graph following consistent color convention.
    BJP_POS -> green
    BJP_NEG -> red
    CONG_POS -> blue
    CONG_NEG -> yellow
    Neutral -> purple
    :param cluster: graph to be plotted
    :param cluster_bjp_neg: BJP negative people to be plotted
    :param cluster_bjp_pos: BJP positive people to be plotted
    :param cluster_con_neg: Congress negative people to be plotted
    :param cluster_con_pos: Congress positive people to be plotted
    :param cluster_neutral: Neutral people to be plotted
    :param count: count for figure id used in plt
    :param graph_name: filename for saving graph
    :return: nothing
    """
    pos = nx.spring_layout(cluster)
    plt.figure(count)
    nx.draw_networkx_nodes(cluster, pos, nodelist=list(cluster_bjp_pos), node_color='green', alpha=.5)
    nx.draw_networkx_nodes(cluster, pos, nodelist=list(cluster_bjp_neg), node_color='red', alpha=.5)
    nx.draw_networkx_nodes(cluster, pos, nodelist=list(cluster_con_pos), node_color='blue', alpha=.5)
    nx.draw_networkx_nodes(cluster, pos, nodelist=list(cluster_con_neg), node_color='yellow', alpha=.5)
    nx.draw_networkx_nodes(cluster, pos, nodelist=list(cluster_neutral), node_color='purple', alpha=.5)
    plt.axis('off')

    nx.draw_networkx_edges(cluster, pos)
    plt.title(title)
    plt.savefig(graph_name)

    plt.show(block=False)


def partition_girvan_newman(sub_graph):
    """
    Uses code from a1 assignment for applying partition_girvan_newman
    :param sub_graph: graph to be partitioned
    :return: list of clusters
    """
    return a1.partition_girvan_newman(sub_graph, 3)


def main():
    clusters = create_graph_clusters()
    print("number of clusters made :", len(clusters))
    print('first partition: cluster 1 has %d nodes %d edges and cluster 2 has %d nodes %d edges' %
          (clusters[0].order(), len(clusters[0].edges), clusters[1].order(), len(clusters[1].edges)))
    draw_clusters(clusters)


def create_graph_clusters():
    graph = create_graph()
    print('Graph has %d nodes %d edges ' %
          (graph.order(), len(graph.edges)))
    sub_graph = a1.get_subgraph(graph, 2)
    print('Created subgraph with min_degree 3, %d nodes, %d edges ' %
          (sub_graph.order(), len(sub_graph.edges)))
    clusters = partition_girvan_newman(sub_graph)
    return clusters


if __name__ == '__main__':
    main()
