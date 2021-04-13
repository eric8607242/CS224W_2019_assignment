import snap
import matplotlib.pyplot as plt
import numpy as np

def cosine_similarity(v1, v2):
    sim_score = np.sum(v1*v2)

    v1_sum = np.sqrt(np.sum(v1*v1))
    v2_sum = np.sqrt(np.sum(v2*v2))

    if v1_sum == 0 or v2_sum == 0:
        return 0
    return sim_score/(v1_sum*v2_sum)

def get_nodes_base_feature(graph):
    node_vector = []

    outdegv = graph.GetNodeOutDegV()
    for i, item in enumerate(outdegv):
        node_id = item.GetVal1()
        degree = item.GetVal2()

        egonet, egnonet_out_edges = graph.GetEgonet(node_id)

        node = graph.GetNI(node_id)
        total_egonet_edge = 0
        for d in range(degree):
            neg_node = graph.GetNI(node.GetNbrNId(d))
            total_egonet_edge += neg_node.GetDeg()

        node_vector.append([degree, (total_egonet_edge-egnonet_out_edges+degree)//2, egnonet_out_edges])

    node_vector = np.array(node_vector)
    return node_vector

def Q1(node_vector, specific_id):
    sim_scores = []
    node_specific_vector = node_vector[specific_id]
    for i, vector in enumerate(node_vector):
        sim_scores.append(cosine_similarity(node_specific_vector, vector))

    sim_scores = np.array(sim_scores)
    argmax_scores = sim_scores.argsort()
    for i, argmax_id in enumerate(reversed(argmax_scores[-6:-1])):
        print("Top-{} node id :{}, cosine similarity:{}".format(i, argmax_id, sim_scores[argmax_id]))

def Q2(graph, node_vector, specific_id=9, K_iteration=2):
    new_node_vector = node_vector
    for k in range(K_iteration):
        sum_node_vector = np.zeros(new_node_vector.shape)
        mean_node_vector = np.zeros(new_node_vector.shape)
        for node_id, vector in enumerate(new_node_vector):
            node = graph.GetNI(node_id)
            degree = node.GetDeg()
            
            neighbor_vectors = []
            for d in range(degree):
                neg_node_id = node.GetNbrNId(d)
                neighbor_vectors.append(new_node_vector[neg_node_id])

            sum_vector = np.sum(neighbor_vectors, axis=0)
            if degree > 0:
                mean_vector = sum_vector/degree
            else:
                mean_vector = np.zeros(sum_vector.shape)


            sum_node_vector[node_id] = sum_vector
            mean_node_vector[node_id] = mean_vector

        new_node_vector = np.concatenate([new_node_vector, mean_node_vector, sum_node_vector], axis=1)


    sim_scores = []
    node_specific_vector = new_node_vector[specific_id]
    for i, vector in enumerate(new_node_vector):
        sim_scores.append(cosine_similarity(node_specific_vector, vector))

    sim_scores = np.array(sim_scores)
    argmax_scores = sim_scores.argsort()
    for i, argmax_id in enumerate(reversed(argmax_scores[-6:-1])):
        print("Top-{} node id :{}, cosine similarity:{}".format(i, argmax_id, sim_scores[argmax_id]))
    return sim_scores
                

def find_node(lower, upper, sim_scores):
    filter_sim_scores = np.array(sim_scores)
    sim_scores_index = np.arange(filter_sim_scores.shape[0])
    sim_scores_index = sim_scores_index[np.logical_and(filter_sim_scores >= lower, filter_sim_scores <= upper)]

    choice = np.random.randint(len(sim_scores_index))

    return sim_scores_index[choice]


def Q3(sim_scores):
    plt.hist(sim_scores)
    plt.show()




if __name__ == "__main__":
    graph = snap.TUNGraph.Load(snap.TFIn("./data/hw1-q2.graph"))

    node_vector = get_nodes_base_feature(graph)
    print("Q1 ------------------------")
    Q1(node_vector, 9)

    print("Q2 ------------------------")
    sim_scores = Q2(graph, node_vector)

    print("Q3 ------------------------")
    Q3(sim_scores)

    first_spot_id = find_node(0.8, 1, sim_scores)
    egonet = graph.GetInEgonetHop(int(first_spot_id), 2)
    labels = {}
    for n in egonet.Nodes():
            labels[n.GetId()] = str(n.GetId())
    egonet.DrawGViz(snap.gvlDot, "./resource/subgraph_1.png", "", labels)


    first_spot_id = find_node(0.00, 0.4, sim_scores)
    egonet = graph.GetInEgonetHop(int(first_spot_id), 2)
    labels = {}
    for n in egonet.Nodes():
            labels[n.GetId()] = str(n.GetId())
    egonet.DrawGViz(snap.gvlDot, "./resource/subgraph_2.png", "", labels)

    first_spot_id = find_node(0.4, 0.8, sim_scores)
    egonet = graph.GetInEgonetHop(int(first_spot_id), 2)
    labels = {}
    for n in egonet.Nodes():
            labels[n.GetId()] = str(n.GetId())
    egonet.DrawGViz(snap.gvlDot, "./resource/subgraph_3.png", "", labels)

    egonet = graph.GetInEgonetHop(9, 2)
    labels = {}
    for n in egonet.Nodes():
            labels[n.GetId()] = str(n.GetId())
    egonet.DrawGViz(snap.gvlDot, "./resource/subgraph_node_9.png", "", labels)
