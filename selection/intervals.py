import numpy as np



class intervals(object):

    r"""
    This class implements methods for intervals or union of two unbounded 
    intervals, when all these sets have a point in their intersection

    """
    def __init__(self, bounded=[], unbounded=[]):
        self.bounded = bounded
        self.unbounded = unbounded

    def intersection(self):
        L = min([ np.inf] + [a for (a,b) in self.unbounded])
        U = max([-np.inf] + [b for (a,b) in self.unbounded])

        l = max([-np.inf] + [a for (a, b) in self.bounded ])
        u = min([ np.inf] + [b for (a, b) in self.bounded ])

        intervs = []
        if (L, U) == (-np.inf, np.inf):
            intervs =  [(l,u)]
        if l < L:
            intervs.append((l, min(L, u)))
        if u > U:
            intervs.append((max(U, l), u))
        return intervs

    def offset(self, off):
        self.bounded = [(a + off, b + off) for a,b in self.bounded ]
        self.unbounded = [(a + off, b + off) for a,b in self.unbounded ]

    def union(self):
        L = max([ np.inf] + [a for (a,b) in self.unbounded])
        U = min([-np.inf] + [b for (a,b) in self.unbounded])

        l = min(a for (a, b) in bounded)
        u = max(b for (a, b) in bounded)

        intervs = []
        if (L, U) == (-np.inf, np.inf):
            intervs =  [(l,u)]
        if l < L:
            intervs.append((l, min(L, u)))
        if u > U:
            intervs.append((max(U, l), u))
        return intervs

    def add(self, I, bounded=True):
        if bounded:
            self.bounded.append(I)
        else:
            self.unbounded.append(I)

