# coding: utf-8

# # CS579: Assignment 1
#
# In this assignment, we'll implement community detection and link prediction algorithms using Facebook "like" data.
#
# The file `edges.txt.gz` indicates like relationships between facebook users. This was collected using snowball sampling: beginning with the user "Bill Gates", I crawled all the people he "likes", then, for each newly discovered user, I crawled all the people they liked.
#
# We'll cluster the resulting graph into communities, as well as recommend friends for Bill Gates.
#
# Complete the **15** methods below that are indicated by `TODO`. I've provided some sample output to help guide your implementation.


# You should not use any imports not listed here:
from collections import deque
from itertools import combinations

import networkx as nx
# import urllib.request


## Community Detection

def example_graph():
    """
    Create the example graph from class. Used for testing.
    Do not modify.
    """
    g = nx.Graph()
    g.add_edges_from(
        [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g


def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.

    You may use these two classes to help with this implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque

    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.

    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node to this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree

    In the doctests below, we first try with max_depth=5, then max_depth=2.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> sorted(node2distances.items())
    [('A', 3), ('B', 2), ('C', 3), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('A', 1), ('B', 1), ('C', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('A', ['B']), ('B', ['D']), ('C', ['B']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 2)
    >>> sorted(node2distances.items())
    [('B', 2), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('B', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('B', ['D']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    """
    ###
    q = deque()

    q.append(('root_parent', root))
    node2parents = dict()
    seen = set()
    path_cost = 0
    node2distances = dict()
    node2num_paths = dict()
    prev_parent = 'root_parent'

    while len(q) > 0 and path_cost <= max_depth:
        pair = q.popleft()
        n = pair[1]
        parent = pair[0]
        if n not in seen:
            if parent != prev_parent:
                path_cost = path_cost + 1
                prev_parent = parent

            seen.add(n)
            if path_cost <= max_depth:
                node2distances[n] = path_cost
                node2num_paths[n] = 1
                node2parents[n] = [parent]
        else:
            all_grand_parents = node2parents[parent]
            all_parents = node2parents[n]
            should_increase = True
            for all_p in all_parents:
                try:
                    if lists_overlap(node2parents[all_p], all_grand_parents):
                        should_increase = False
                        break
                except:
                    pass
            temp_path = (path_cost, path_cost + 1)[should_increase]
            if node2distances[n] == temp_path and parent not in all_parents:
                node2num_paths[n] = node2num_paths[n] + 1
                all_parents.append(parent)

        for node_neighbors in graph.neighbors(n):
            if node_neighbors not in seen:
                q.append((n, node_neighbors))

    node2parents_copy = node2parents.copy()
    for l in node2parents.keys():
        if l not in node2distances:
            node2parents_copy.pop(l)
    node2parents_copy.pop(root)
    return node2distances, node2num_paths, node2parents_copy


def lists_overlap(a, b):
    return bool(set(a) & set(b))


def complexity_of_bfs(V, E, K):
    """
    If V is the number of vertices in a graph, E is the number of
    edges, and K is the max_depth of our approximate breadth-first
    search algorithm, then what is the *worst-case* run-time of
    this algorithm? As usual in complexity analysis, you can ignore
    any constant factors. E.g., if you think the answer is 2V * E + 3log(K),
    you would return V * E + math.log(K)
    >>> v = complexity_of_bfs(13, 23, 7)
    >>> type(v) == int or type(v) == float
    True
    """
    '''
    O(1) operation for each vertex - total O(V) i.e. popping element 
    O(1) operation for each Edge - total O(V) i.e. adding element 
    Summing these operation we have O(V) + O(E) 
    for sparse graphs, where E = O(V), we have overall O(V). 
    While for dense graphs where we have E=O(V^2), we get overall O(E).

    And O(V+E)=O(max(V,E)). 
     
    '''

    if V > E:
        return V
    else:
        return E

    # return (E, V)[V > E]


def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.
    See p 352 From your text:
    https://github.com/iit-cs579/main/blob/master/read/lru-10.pdf
        The third and final step is to calculate for each edge e the sum
        over all nodes Y of the fraction of shortest paths from the root
        X to Y that go through e. This calculation involves computing this
        sum for both nodes and edges, from the bottom. Each node other
        than the root is given a credit of 1, representing the shortest
        path to that node. This credit may be divided among nodes and
        edges above, since there could be several different shortest paths
        to the node. The rules for the calculation are as follows: ...

    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

      Any edges excluded from the results in bfs should also be exluded here.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> result = bottom_up('E', node2distances, node2num_paths, node2parents)
    >>> sorted(result.items())
    [(('A', 'B'), 1.0), (('B', 'C'), 1.0), (('B', 'D'), 3.0), (('D', 'E'), 4.5), (('D', 'G'), 0.5), (('E', 'F'), 1.5), (('F', 'G'), 0.5)]
    """

    bottom_up_dict = dict()
    for node, parents in node2parents.items():
        my_node_children_list = find_children(node, node2parents)
        my_node_parents = node2parents[node]
        my_parent_proportions = credit_proportion_for_parents(my_node_parents, node2num_paths)
        my_credit = recurse_on_child(my_node_children_list, bottom_up_dict, node2parents, node2num_paths, node)
        for p in my_node_parents:
            my_node_parent_edge = tuple((node, p))
            rounded_credit = round(my_parent_proportions[p] * my_credit, 1)
            bottom_up_dict[my_node_parent_edge] = rounded_credit
    return bottom_up_dict


def recurse_on_child(child_list, bottom_up_dict, node2parents, node2num_paths, child):
    # type: (list, dict, dict, dict, node) -> int
    my_credit = 1
    if len(child_list) != 0:
        for c in child_list:
            child_edge = tuple((child, c))
            if bottom_up_dict.get(child_edge, -1) == -1:
                child_node_children_list = find_children(c, node2parents)
                child_node_parents = node2parents[c]
                child_parent_proportions = credit_proportion_for_parents(child_node_parents, node2num_paths)
                child_credit = child_parent_proportions[child] * recurse_on_child(child_node_children_list,
                                                                                  bottom_up_dict,
                                                                                  node2parents, node2num_paths, c)
                rounded_credit = round(child_credit, 1)
                bottom_up_dict[child_edge] = rounded_credit
                my_credit += rounded_credit
            else:
                my_credit = my_credit + bottom_up_dict[child_edge]

    return round(my_credit, 1)


def find_children(node, node2parents):
    list_of_children = []
    for find_node, find_parent in node2parents.items():
        if node in find_parent:
            list_of_children.append(find_node)
    return list_of_children


def credit_proportion_for_parents(parents, node2num_paths):
    list_of_shortest_path = []
    parent_proportion_dict = dict()
    for p in parents:
        list_of_shortest_path.append(node2num_paths[p])
    for p in parents:
        p_proportion = node2num_paths[p] / float(sum(list_of_shortest_path))
        parent_proportion_dict[p] = p_proportion
    return parent_proportion_dict


def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.

    You should call the bfs and bottom_up functions defined above for each node
    in the graph, and sum together the results. Be sure to divide by 2 at the
    end to get the final betweenness.

    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A dict mapping edges to betweenness. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).

    >>> sorted(approximate_betweenness(example_graph(), 2).items())
    [(('A', 'B'), 2.0), (('A', 'C'), 1.0), (('B', 'C'), 2.0), (('B', 'D'), 6.0), (('D', 'E'), 2.5), (('D', 'F'), 2.0), (('D', 'G'), 2.5), (('E', 'F'), 1.5), (('F', 'G'), 1.5)]
    """

    approx_betweenness = dict()

    for node in graph.nodes():
        node2distances, node2num_paths, node2parents = bfs(graph, node, max_depth)
        temp_dict = bottom_up(node, node2distances, node2num_paths, node2parents)
        for edge, betweenness in temp_dict.items():
            current_bet = approx_betweenness.get(edge, 0)
            approx_betweenness[edge] = round(current_bet, 1) + round(betweenness, 1)

    for edge, betweenness in approx_betweenness.items():
        approx_betweenness[edge] = round(betweenness / 2, 1)

    return approx_betweenness


def get_components(graph):
    """
    A helper function you may use below.
    Returns the list of all connected components in the given graph.
    """
    return [c for c in nx.connected_component_subgraphs(graph)]


def partition_girvan_newman(graph, max_depth):
    """
    Use your approximate_betweenness implementation to partition a graph.
    Unlike in class, here you will not implement this recursively. Instead,
    just remove edges until more than one component is created, then return
    those components.
    That is, compute the approximate betweenness of all edges, and remove
    them until multiple components are created.

    You only need to compute the betweenness once.
    If there are ties in edge betweenness, break by edge name (e.g.,
    (('A', 'B'), 1.0) comes before (('B', 'C'), 1.0)).

    Note: the original graph variable should not be modified. Instead,
    make a copy of the original graph prior to removing edges.
    See the Graph.copy method https://networkx.github.io/documentation/stable/reference/classes/generated/networkx.Graph.copy.html
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A list of networkx Graph objects, one per partition.

    >>> components = partition_girvan_newman(example_graph(), 5)
    >>> components = sorted(components, key=lambda x: sorted(x.nodes())[0])
    >>> sorted(components[0].nodes())
    ['A', 'B', 'C']
    >>> sorted(components[1].nodes())
    ['D', 'E', 'F', 'G']
    """

    components = []
    graph_copy = graph.copy()

    while not len(components) > 1:
        betweenness = approximate_betweenness(graph_copy, max_depth)
        edge_to_remove = max(betweenness, key=betweenness.get)
        graph_copy.remove_edge(edge_to_remove[0], edge_to_remove[1])
        components = get_components(graph_copy)

    return components


def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.

    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
    """

    graph_copy = graph.copy()
    for node in graph.nodes():
        if graph.degree(node) < min_degree:
            graph_copy.remove_node(node)

    return graph_copy


""""
Compute the normalized cut for each discovered cluster.
I've broken this down into the three next methods.
"""


def volume(nodes, graph):
    """
    Compute the volume for a list of nodes, which
    is the number of edges in `graph` with at least one end in
    nodes.
    Params:
      nodes...a list of strings for the nodes to compute the volume of.
      graph...a networkx graph

    >>> volume(['A', 'B', 'C'], example_graph())
    4
    """
    count = 0
    for e in graph.edges():
        if e[0] in nodes or e[1] in nodes:
            count += 1
    return count


def cut(S, T, graph):
    """
    Compute the cut-set of the cut (S,T), which is
    the set of edges that have one endpoint in S and
    the other in T.
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An int representing the cut-set.

    >>> cut(['A', 'B', 'C'], ['D', 'E', 'F', 'G'], example_graph())
    1
    """

    count = 0
    for e in graph.edges():
        if (e[0] in S and e[1] in T) or (e[0] in T and e[1] in S):
            count = count + 1
    return count


def norm_cut(S, T, graph):
    """
    The normalized cut value for the cut S/T. (See lec06.)
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An float representing the normalized cut value
    >>> norm_cut(['A', 'B', 'C'], ['D', 'E', 'F', 'G'], example_graph())
    0.41666666666666663

    """

    cut_value = cut(S, T, graph)
    vol_s = volume(S, graph)
    vol_t = volume(T, graph)
    normalized_cut = (cut_value / float(vol_s)) + (cut_value / float(vol_t))
    return normalized_cut


def brute_force_norm_cut(graph, max_size):
    """
    Enumerate over all possible cuts of the graph, up to max_size, and compute the norm cut score.
    Params:
        graph......graph to be partitioned
        max_size...maximum number of edges to consider for each cut.
                   E.graph_copy, if max_size=2, consider removing edge sets
                   of size 1 or 2 edges.
    Returns:
        (unsorted) list of (score, edge_list) tuples, where
        score is the norm_cut score for each cut, and edge_list
        is the list of edges (source, target) for each cut.
        

    Note: only return entries if removing the edges results in exactly
    two connected components.

    You may find itertools.combinations useful here.

    >>> r = brute_force_norm_cut(example_graph(), 1)
    >>> len(r)
    1
    >>> r
    [(0.41666666666666663, [('B', 'D')])]
    >>> r = brute_force_norm_cut(example_graph(), 2)
    >>> len(r)
    14
    >>> sorted(r)[0]
    (0.41666666666666663, [('A', 'B'), ('B', 'D')])
    """
    max_sub_components = 2
    graph_copy = graph.copy()
    list_of_cut_score_ebunch = []

    for i in range(0, max_size):
        for ebunch in combinations(graph.edges(), i + 1):
            ebunch = list(ebunch)
            graph_copy.remove_edges_from(ebunch)
            sub_components = get_components(graph_copy)

            if len(sub_components) == max_sub_components:
                norm_cut_value = norm_cut(sub_components[0], sub_components[1], graph)
                list_of_cut_score_ebunch.append((norm_cut_value, ebunch))

            graph_copy.add_edges_from(ebunch)

    return list_of_cut_score_ebunch


def score_max_depths(graph, max_depths):
    """
    In order to assess the quality of the approximate partitioning method
    we've developed, we will run it with different values for max_depth
    and see how it affects the norm_cut score of the resulting partitions.
    Recall that smaller norm_cut scores correspond to better partitions.

    Params:
      graph........a networkx Graph
      max_depths...a list of ints for the max_depth values to be passed
                   to calls to partition_girvan_newman

    Returns:
      A list of (int, float) tuples representing the max_depth and the
      norm_cut value obtained by the partitions returned by
      partition_girvan_newman. See Log.txt for an example.
    """

    list_of_depth_cut_score = []
    for depth in max_depths:
        girvan_newman_partition = partition_girvan_newman(graph, depth)
        list_of_depth_cut_score.append((depth, norm_cut(girvan_newman_partition[0], girvan_newman_partition[1], graph)))

    return list_of_depth_cut_score


## Link prediction

# Next, we'll consider the link prediction problem. In particular,
# we will remove 5 of the accounts that Bill Gates likes and
# compute our accuracy at recovering those links.

def make_training_graph(graph, test_node, n):
    """
    To make a training graph, we need to remove n edges from the graph.
    As in lecture, we'll assume there is a test_node for which we will
    remove some edges. Remove the edges to the first n neighbors of
    test_node, where the neighbors are sorted alphabetically.
    E.g., if 'A' has neighbors 'B' and 'C', and n=1, then the edge
    ('A', 'B') will be removed.

    Be sure to *copy* the input graph prior to removing edges.

    Params:
      graph.......a networkx Graph
      test_node...a string representing one node in the graph whose
                  edges will be removed.
      n...........the number of edges to remove.

    Returns:
      A *new* networkx Graph with n edges removed.

    In this doctest, we remove edges for two friends of D:
    >>> g = example_graph()
    >>> sorted(g.neighbors('D'))
    ['B', 'E', 'F', 'G']
    >>> train_graph = make_training_graph(g, 'D', 2)
    >>> sorted(train_graph.neighbors('D'))
    ['F', 'G']
    """

    graph_copy = graph.copy()
    count = 0
    for node_neighbor in sorted(graph.neighbors(test_node)):
        if count < n:
            graph_copy.remove_edge(node_neighbor, test_node)
            count = count + 1
    return graph_copy


def jaccard(graph, node, k):
    """
    Compute the k highest scoring edges to add to this node based on
    the Jaccard similarity measure.
    Note that we don't return scores for edges that already appear in the graph.

    Params:
      graph....a networkx graph
      node.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.

    Returns:
      A list of tuples in descending order of score representing the
      recommended new edges. Ties are broken by
      alphabetical order of the terminal node in the edge.

    In this example below, we remove edges (D, B) and (D, E) from the
    example graph. The top two edges to add according to Jaccard are
    (D, E), with score 0.5, and (D, A), with score 0. (Note that all the
    other remaining edges have score 0, but 'A' is first alphabetically.)

    >>> g = example_graph()
    >>> train_graph = make_training_graph(g, 'D', 2)
    >>> jaccard(train_graph, 'D', 2)
    [(('D', 'E'), 0.5), (('D', 'A'), 0.0)]
    """
    graph_neighbors = set(graph.neighbors(node))
    list_nodes = []
    for a in graph.nodes():
        if a not in graph_neighbors and a != node:
            list_nodes.append((node, a))
    jc_score = nx.jaccard_coefficient(graph, list_nodes)
    # for u, v, p in jc_score:
    #  '(%s, %s) -> %.8f' % (u, v, p)

    return list(map(lambda x: ((x[0], x[1]), x[2]), sorted(jc_score, key=lambda x: float(x[2]), reverse=True)[:k]))


def evaluate(predicted_edges, graph):
    """
    Return the fraction of the predicted edges that exist in the graph.

    Args:
      predicted_edges...a list of edges (tuples) that are predicted to
                        exist in this graph
      graph.............a networkx Graph

    Returns:
      The fraction of edges in predicted_edges that exist in the graph.

    In this doctest, the edge ('D', 'E') appears in the example_graph,
    but ('D', 'A') does not, so 1/2 = 0.5

    >>> evaluate([('D', 'E'), ('D', 'A')], example_graph())
    0.5
    """
    count = 0
    for a in predicted_edges:
        if a in graph.edges():
            count = count + 1
    prob = count / float(len(predicted_edges))
    return prob


"""
Next, we'll download a real dataset to see how our algorithm performs.
"""


def download_data():
    """
    Download the data. Done for you.
    """
    urllib.request.urlretrieve('http://cs.iit.edu/~culotta/cs579/a1/edges.txt.gz', 'edges.txt.gz')


def read_graph():
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    Done for you.
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')


def main():
    """
    FYI: This takes ~10-15 seconds to run on my laptop.
    """
    download_data()
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1, 5)))
    clusters = partition_girvan_newman(subgraph, 3)
    print('%d clusters' % len(clusters))
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('smaller cluster nodes:')
    print(sorted(clusters, key=lambda x: x.order())[0].nodes())
    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))

    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
          evaluate([x[0] for x in jaccard_scores], subgraph))


if __name__ == '__main__':
    main()
