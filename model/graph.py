"""
Graph adjacency construction for ST-GCN.

Ported from OpenHands (AI4Bharat/OpenHands), openhands/models/encoder/graph/graph_utils.py,
with the same computations, so the graph and its spatial partitioning match OpenHands' ST-GCN.
"""
import numpy as np


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn)


class Graph:
    """
    Builds the (K, V, V) partitioned adjacency tensor used by ST-GCN's spatial graph
    convolution, following Yan et al. 2018 (ST-GCN) / OpenHands' `GraphWithPartition`.

    Args:
        num_nodes (int): number of joints V.
        center (int): index of the root/center joint used by the "spatial" partitioning
            strategy to split neighbours into "closer to center" / "further from center".
        inward_edges (list[[int, int]]): directed edges pointing from an outer joint
            towards the center (child -> parent). Self-edges are added automatically.
        strategy (str): "uniform" | "distance" | "spatial". OpenHands uses "spatial".
        max_hop (int): maximum neighbourhood hop distance to consider. Default 1.
        dilation (int): hop dilation. Default 1.
    """

    def __init__(self, num_nodes, center, inward_edges, strategy="spatial", max_hop=1, dilation=1):
        self.num_nodes = num_nodes
        self.center = center
        self.self_edges = [[i, i] for i in range(self.num_nodes)]
        self.inward_edges = inward_edges
        self.edges = self.self_edges + self.inward_edges

        self.max_hop = max_hop
        self.dilation = dilation

        self.hop_dis = get_hop_distance(self.num_nodes, self.edges, max_hop=max_hop)
        self.get_adjacency(strategy)

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_nodes, self.num_nodes))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == "uniform":
            A = np.zeros((1, self.num_nodes, self.num_nodes))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == "distance":
            A = np.zeros((len(valid_hop), self.num_nodes, self.num_nodes))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == "spatial":
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_nodes, self.num_nodes))
                a_close = np.zeros((self.num_nodes, self.num_nodes))
                a_further = np.zeros((self.num_nodes, self.num_nodes))
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            self.A = np.stack(A)
        else:
            raise ValueError(f"Graph construction strategy '{strategy}' is not supported")
