import snap
import numpy as np
import matplotlib.pyplot as plt

def Q1(wiki_graph):
    num_node = wiki_graph.GetNodes()
    num_self_edge = wiki_graph.CntSelfEdges()
    num_dir_edge = wiki_graph.CntUniqDirEdges()
    num_undir_edge = wiki_graph.CntUniqUndirEdges()
    num_reciprocated_edge = num_dir_edge - num_undir_edge
    num_zero_outdegree_node = wiki_graph.CntOutDegNodes(0)
    num_zero_indegree_node = wiki_graph.CntInDegNodes(0)

    DegToCntV = wiki_graph.GetOutDegCnt()
    num_outdegree_g_10_node = 0
    for item in DegToCntV:
        if item.GetVal1() > 10:
            num_outdegree_g_10_node += item.GetVal2()

    DegToCntV = wiki_graph.GetInDegCnt()
    num_indegree_l_10_node = 0
    for item in DegToCntV:
        if item.GetVal1() < 10:
            num_indegree_l_10_node += item.GetVal2()

    print(f"The number of nodes in the network. : {num_node}")
    print(f"The number of nodes with a self-edge. : {num_self_edge}")
    print(f"The number of directed edges. : {num_dir_edge}")
    print(f"The number of undirected edges. : {num_undir_edge}")
    print(f"The number of reciprocated edges. : {num_reciprocated_edge}")
    print(f"The number of nodes of zero out-degree. : {num_zero_outdegree_node}")
    print(f"The number of nodes of zero in-degree. : {num_zero_indegree_node}")
    print(f"The number of nodes with mode than 10 out-degree. : {num_outdegree_g_10_node}")
    print(f"The number of nodes with fewer than 10 in-degree. : {num_indegree_l_10_node}")



def Q2(wiki_graph):
    DegToCntV = wiki_graph.GetOutDegCnt()
    degree_list = []
    num_list = []
    for item in DegToCntV:
        degree_list.append(item.GetVal1())
        num_list.append(item.GetVal2())

    degree_list = np.array(degree_list)
    num_list = np.array(num_list)


    degree_list = np.log10(degree_list)[1:]
    num_list = np.log10(num_list)[1:]

    a, b = np.polyfit(degree_list, num_list, 1)
    f = np.poly1d([a, b])

    plt.plot(degree_list, num_list)
    plt.plot(degree_list, f(degree_list))
    plt.show()


def Q3(stack_graph):
    num_wcc = stack_graph.GetWccs().Len()

    wcc = stack_graph.GetMxWcc()
    num_edge_in_wcc = wcc.GetNodes()
    num_node_in_wcc = wcc.GetEdges()

    pagerank = stack_graph.GetPageRank()
    pagerank_scores = []
    pagerank_node = []
    for item in pagerank:
        pagerank_scores.append(pagerank[item])
        pagerank_node.append(item)

    pagerank_scores = np.array(pagerank_scores)
    pagerank_node = np.array(pagerank_node)
    top3_page_rank_node = pagerank_node[pagerank_scores.argsort()[-3:]]

    auth_scores = []
    hub_scores = []
    hits_node = []
    NIdHubH, NIdAuthH = stack_graph.GetHits()
    for item_hub, item_auth in zip(NIdHubH, NIdAuthH):
        auth_scores.append(NIdAuthH[item_auth])
        hub_scores.append(NIdHubH[item_hub])
        hits_node.append(item_hub)

    auth_scores = np.array(auth_scores)
    hub_scores = np.array(hub_scores)
    hits_node = np.array(hits_node)

    top3_auth = hits_node[auth_scores.argsort()[-3:]]
    top3_hub = hits_node[hub_scores.argsort()[-3:]]


    print(f"The number of weakly connected components in the network. : {num_wcc}")
    print(f"The number of edges in the largest weakly connected component. : {num_edge_in_wcc}")
    print(f"The number of nodes in the largest weakly connected component. : {num_node_in_wcc}")
    print(f"The top 3 most central nodes by PageRank scores. : {top3_page_rank_node}")
    print(f"The top3 hubs by HITS scores. : {top3_hub}")
    print(f"The top3 auth by HITS scores. : {top3_auth}")


if __name__ == "__main__":
    wiki_graph = snap.LoadEdgeList(snap.TNGraph, "./data/wiki-Vote.txt", 0, 1)

    print("Q1 ----------------------------------------------")
    Q1(wiki_graph)
    print("Q2 ----------------------------------------------")
    Q2(wiki_graph)

    print("Q3 ----------------------------------------------")
    stack_graph = snap.LoadEdgeList(snap.TNGraph, "./data/stackoverflow-Java.txt", 0, 1)
    Q3(stack_graph)
