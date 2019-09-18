import numpy as np
import mpmath as mp

from abc import ABCMeta, abstractmethod

from .intervals import intervals

from ..truncated.api import truncated_chi, truncated_chi2
from scipy.stats import norm

class constraints(object):
    ## Means that constraint is a Meta Class
    __metaclass__ = ABCMeta
    
    ## These two fonctions have to be defined for each subclass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def bounds(self, nu, y):
        pass

    def __invert__(self):
        inverse = cons_op.cons_op.complement(self)
        return inverse
    

class cons_op(constraints):

    def __init__(self):
        self._cons_list = []
        self._op = None


    @staticmethod
    def intersection(*cons):
        if not all(isinstance(c, constraints) for c in cons):
            t = type([c for c in cons if not isinstance(c, constraints)][0])
            raise TypeError("Not a constraint : "+ repr(t))

        cons = [c for c in cons if not isinstance(c, noConstraint)]
        intersection = cons_op()
        intersection._cons_list = list(cons)
        intersection._op = 'I'
        return intersection

    @staticmethod
    def union(*cons):
        if not all(isinstance(c, constraints) for c in cons):
            t = type([c for c in cons if not isinstance(c, constraints)][0])
            raise TypeError("Not a constraint : "+ repr(t))
        union = cons_op()
        union._cons_list = list(cons)
        union._op = 'U'
        return union

    @staticmethod
    def complement(cons):
        if isinstance(cons, cons_op):
            op = cons._op
            if op == 'U':
                cons_gen = (~cons for cons in cons._cons_list)
                complement = cons_op.intersection(*cons_gen)
            elif op == 'I':
                cons_gen = (~cons for cons in cons._cons_list)
                complement = cons_op.union(*cons_gen)
            elif op == 'C':
                complement = cons._cons_list[0]

        else:
            complement = cons_op()
            complement._cons_list = [cons]
            complement._op = 'C'

        return complement


    def __call__(self, y):
        if self._op == 'U':
            b = any(cons(y) for cons in self._cons_list)
        elif self._op == 'I':
            b = all(cons(y) for cons in self._cons_list)
        elif self._op == 'C':
            b = not self._cons_list[0](y)
        return b
    
        
    def bounds(self, nu, y):
        if not self(y):
            raise ValueError("y does not satisfy the constraints")

        interv_gen = [cons.bounds(nu, y) for cons in self._cons_list]
            
        if self._op == 'U':
            I = intervals.union(*interv_gen)
        elif self._op == 'I':
            I = intervals.intersection(*interv_gen)
        elif self._op == 'C':
            I = ~(interv_gen[0])
        return I




class noConstraint(constraints):
    def __init__(self):
        pass

    def __call__(self, y):
        return True

    def bounds(self, nu, y):
        I = ~intervals()
        return I


