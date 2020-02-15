# Python3 implementation to find minimum
# spanning tree for adjacency representation.
from sys import maxsize
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components


INT_MAX = maxsize


# Returns true if edge u-v is a valid edge to be
# include in MST. An edge is valid if one end is
# already included in MST and other is not in MST.
def isValidEdge(u, v, inMST):
    if u == v:
        return False
    if inMST[u] == False and inMST[v] == False:
        return False
    elif inMST[u] == True and inMST[v] == True:
        return False
    return True

def primMst(graph, root_index):
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

def prim_for_connected_graph(graph, root_index):
    inMST = np.full((graph.shape[0],1), False)
    adjacency_matrix = lil_matrix(graph.shape, dtype=np.bool)
    # Include first vertex in MST
    inMST[root_index] = True
    # Keep adding edges while number of included
    # edges does not become V-1.
    edge_count = 0
    mincost = 0
    V = graph.shape[0]
    while edge_count < V - 1:
        # Find minimum weight valid edge.
        minn = INT_MAX
        a = -1
        b = -1
        for i in range(V):
            for j in range(V):
                if graph[i, j] < minn:
                    if isValidEdge(i, j, inMST) and graph[i,j] > 0:
                        minn = graph[i,j]
                        a = i
                        b = j

        if a != -1 and b != -1:
            edge_count += 1
            mincost += minn
            adjacency_matrix[a,b] = True
            adjacency_matrix[b, a] = True
            inMST[b] = inMST[a] = True
    adjacency_matrix = adjacency_matrix.tocsr()
    print(adjacency_matrix)
    print("Minimum cost = %d" % mincost)


# Driver Code
if __name__ == "__main__":
    ''' Let us create the following graph 
        2 3 
    (0)--(1)--(2) 
    | / \ | 
    6| 8/ \5 |7 
    | /	 \ | 
    (3)-------(4) 
            9		 '''

    cost = csr_matrix([[0, 8, 0, 0],
[8, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0]])

    # Print the solution
    primMST(cost, 1)

# This code is contributed by
# sanjeev2552
