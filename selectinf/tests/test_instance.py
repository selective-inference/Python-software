from numpy import inf
from itertools import product
from .instance import gaussian_instance, logistic_instance, HIV_NRTI

def test_gaussian_instance():

    for scale, center, random_signs, df in product(
        [True, False],
        [True, False],
        [True, False],
        [40, inf]):
        gaussian_instance(n=10,
                          p=20,
                          s=4,
                          random_signs=random_signs,
                          scale=scale,
                          center=center,
                          df=df)

def test_logistic_instance():

    for scale, center, random_signs in product(
        [True, False],
        [True, False],
        [True, False]):
        logistic_instance(n=10,
                          p=20,
                          s=4,
                          random_signs=random_signs,
                          scale=scale,
                          center=center)

def test_HIV_instance():

    HIV_NRTI()


    
