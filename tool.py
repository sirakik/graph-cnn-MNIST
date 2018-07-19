import numpy as np

def generate_adjacency_matrix(size):
    adj_side = size * size #784
    adj = np.zeros((adj_side,adj_side)) #(784,784)

    for y in range(size):
        for x in range(size):
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    if (y + i) in range(size) and (x + j) in range(size):
                        adj[y * size + x, y * size + x + i * size + j] = 1
                    else:
                        continue

    norm_adj = normalize_digraph(adj)
    return norm_adj

def normalize_digraph(adj):  # 0~1に正規化された隣接行列
    Dl = np.sum(adj, 0)  # 各関節ごとの接続数リスト[4. 4. 4. 3. 2. ...]
    num_node = adj.shape[0]  # 784
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    #AD = np.dot(adj,Dn)
    AD = np.dot(Dn,adj)
    return AD


