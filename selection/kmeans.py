import numpy as np

def swap_matrix(config, swap):
    """
    Take a configure of 2-means,
    make a swap of one point to the other class and
    compute the rank 2 difference between their projection matrices.
    """
    new_config = config.copy()
    if np.all(new_config[swap] == [1,0]):
        new_config[swap] = [0,1]
    else:
        new_config[swap] = [1,0]
    P1 = np.dot(config, config.T)
    P2 = np.dot(new_config, new_config.T)
    U, D, V = np.linalg.svd(P1 - P2, full_matrices=0)
    V = V[:2]
    print D
    print np.linalg.eigvalsh(P1-P2)
    return new_config, V, P1, P2

config = np.array([[0,0,0,0,1,1,1,1,1],[1,1,1,1,0,0,0,0,0]]).T

new_config, V, P1, P2 = swap_matrix(config, 3)

Y = np.random.standard_normal((config.shape[0], 3))
D1 = np.diag(np.dot(Y.T, np.dot(P1, Y))).sum()
D2 = np.diag(np.dot(Y.T, np.dot(P2, Y))).sum()
print np.linalg.norm(np.dot(V[0], Y)), np.linalg.norm(np.dot(V[1], Y))
