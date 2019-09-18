import numpy as np
import warnings
import operator

from heapq import merge

class intervals(object):

    r"""
    This class implements methods for intervals or union of two unbounded 
    intervals, when all these sets have a point in their intersection

    """
    def __init__(self, I = None):
        """
        Create a intervals object, with some unbounded and bounded intervals

        Parameters
        ----------
        I : tuple
              I is a tuple (inf, sup), the interval created

        Returns
        -------
        interv : intervals
              The intervals object

        Warning : sup has to be larger than inf. If not, it raises a 
        ValueError exception
        If sup == inf, it creates an empty interval, and raise a Warning


        >>> I = intervals()
        >>> I2 = intervals((-1, 1))
        """

        if I == None:
            self._U = []

        else:
            ## Check that the interval is correct
            (inf, sup) = I
            if sup < inf:
                raise ValueError("The given tuple " + \
                                 "does not represent an interval : " + repr(I))

            # elif inf == sup:
            #     self._U = []

            else:
                self._U = [I]


    def __call__(self, x):
        """
        Check if x is in the intersection of the intervals

        Parameters
        ----------
        x : float
              The point you want to know if it is in the intervals

        Returns
        -------
        is_in : bool
              True if x is in the intersection, False if it's not

        Examples
        --------
        
        >>> I = intervals()
        >>> I(2)
        False
        >>> I = intervals.intersection(intervals((-1, 6)), \
                                       intervals(( 0, 7)), \
                                       ~intervals((1, 4)))
        >>> x1, x2, x3, x4, x5 = 0.5, 1.5, 5, 6.5, 8
        >>> I(x1), I(x2), I(x3), I(x4), I(x5)
        (True, False, True, False, False)
        """
        return any( a <= x and x <= b for (a, b) in self )


    def __len__(self):
        """
        Return the number of connex intervas composing this instance

        >>> I = intervals.intersection(intervals((-1, 6)), \
                                       intervals(( 0, 7)), \
                                       ~intervals((1, 4)))
        
        >>> len(I)
        2
        """
        return len(self._U)

    def __invert__(self):
        """
        Return the complement of the interval in the reals

        >>> I = intervals.intersection(intervals((-1, 6)), \
                                       intervals(( 0, 7)), \
                                       ~intervals((1, 4)))
        >>> print(~I)
        [(-inf, 0), (1, 4), (6, inf)]
        """

        if len(self) == 0:
            return intervals((-np.inf, np.inf))

        inverse = intervals()
        a, _ = self._U[0]
        if a > -np.inf:
            inverse._U.append((-np.inf, a))
            
        for (a1, b1), (a2, b2) in zip(self._U[:-1], self._U[1:]):
            inverse._U.append((b1, a2))
                
        _, b = self._U[-1]
        if b < np.inf:
            inverse._U.append((b, np.inf))

        return inverse


    def __repr__(self):
        return repr(self._U)

    def __iter__(self):
        return iter(self._U)

    def __getitem__(self,index):
        return self._U[index]

       
    @staticmethod
    def union(*interv):
        """
        Return the union of all the given intervals

        Parameters
        ----------
        interv1, ... : interv
              intervals instance

        Returns
        -------
        union, a new intervals instance, representing the union of interv1, ...

        >>> I = intervals.union(intervals((-np.inf, 0)), \
                                intervals((-1, 1)), \
                                intervals((3, 6)))
        >>> print(I)
        [(-inf, 1), (3, 6)]
        """
        ## Define the union of an empty family as an empty set

        union = intervals()
        if len(interv) == 0:
            return interv

        interv_merged_gen = merge(*interv)

        old_a, old_b = None, None
        for new_a, new_b in interv_merged_gen:
            if old_b is not None and new_a < old_b: # check to see if union of (old_a, old_b) and 
                                                    # (new_a, new_b) is (old_a, new_b) 
                old_b = max(old_b, new_b)
            elif old_b is None: # first interval
                old_a, old_b = new_a, new_b
            else:
                union._U.append((old_a, old_b))
                old_a, old_b = new_a, new_b

        union._U.append((old_a, old_b))

        return union

    @staticmethod
    def intersection(*interv):
        """
        Return the intersection of all the given intervals

        Parameters
        ----------
        interv1, ... : interv
              intervals instance

        Returns
        -------
        intersection, a new intervals instance, representing the intersection
        of interv1, ...

        >>> I = intervals.intersection(intervals((-1, 6)), \
                                       intervals(( 0, 7)), \
                                       ~intervals((1, 4)))
        >>> print(I)
        [(0, 1), (4, 6)]
        
        """
        if len(interv) == 0:
            I = intervals()
            return ~I
        return ~(intervals.union(*(~I for I in interv)))

    def __add__(self, offset):
        """
        Add an offset to the intervals

        Parameters
        ----------
        off : float
              The offset added

        Returns
        -------
        interv : intervals
              a new instance, self + offset

        Examples
        --------
        >>> I = intervals.intersection(intervals((-1, 6)), \
                                       intervals(( 0, 7)), \
                                       ~intervals((1, 4)))
        >>> J = I+2
        >>> print(J)
        [(2, 3), (6, 8)]

        """
        interv = intervals()
        interv._U = [(a+offset, b+offset) for (a, b) in self._U]
        return interv 



if __name__ == "__main__":
    import doctest
    doctest.testmod()

