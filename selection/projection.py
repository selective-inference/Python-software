import numpy as np

class projection(object):

    """
    A projection matrix, U an orthonormal basis of the column space.
    
    Warning: we do not check if U has orthonormal columns. 

    This can be enforced by calling the `orthogonalize` method
    which returns a new instance.
    
    """
    def __init__(self, U):
        self.U = U

    def __call__(self, Z, rank=None):
        if rank is None:
            return np.dot(self.U, np.dot(self.U.T, Z))
        else:
            return np.dot(self.U[:,:rank], np.dot(self.U.T[:rank], Z))

    def stack(self, Unew):
        """
        Form a new projection matrix by hstack.

        Warning: no check is mode to ensure U has orthonormal columns.
        """
        return projection(np.hstack([self.U, Unew]))

    def orthogonalize(self):
        """
        Force columns to be orthonormal.

        >>> M = np.array([[2., 1., -14.], [0., 1., -1.], [0., 1., -1.]])
        >>> P = projection(M)
        >>> P = P.orthogonalize()
        >>> v = np.array([[1., 2., 3.]])
        >>> P(v.T).T
        array([[ 1. ,  2.5,  2.5]])

        """
        U, D, V = np.linalg.svd(self.U, full_matrices=False)
        tol = D.max() * max(D.shape) * np.finfo(D.dtype).eps
        U = U[:, np.fabs(D) > tol ]
        return projection(U)



