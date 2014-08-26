
import numpy as np
cimport numpy as np

#from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free


cdef extern from "preparation_Eig_Vect.h":
    void samples(int n,
                 int dim,
                 int seed,
                 double* initial,
                 int numlin,
                 int numquad,
                 double* lin,
                 double* quad,
                 double* quad_lin,
                 double* offset_lin,
                 double* offset_quad,
                 double* samples_Carray)



def quad_sampler(int n_sample, 
                 initial,
                 quad,# = np.array([]).reshape((0, 0, 0)), 
                 quad_lin,# = np.array([]).reshape((0, 0)), 
                 lin,# = np.array([]).reshape((0,0)), 
                 offset_quad,# = np.array([]), 
                 offset_lin # = np.array([]) 
                 ):

    

    cdef int numquad = quad.shape[0]
    cdef int p = quad.shape[1]
    cdef int numlin = lin.shape[0]

    cdef np.ndarray[np.double_t, ndim=3] quad2 = np.ascontiguousarray(-quad)
    cdef np.ndarray[np.double_t, ndim=2] quad_lin2 = np.ascontiguousarray(-quad_lin)
    cdef np.ndarray[np.double_t, ndim=1] offset_quad2 = np.ascontiguousarray(offset_quad)

    cdef double *pt_quad
    cdef double *pt_quad_lin
    cdef double *pt_quad_offset
    if numquad > 0:
        pt_quad = &quad2[0, 0, 0]
        pt_quad_lin = &quad_lin2[0, 0]
        pt_quad_offset = &offset_quad2[0]
        


    print "quad inequalities generated"

    
    cdef np.ndarray[np.double_t, ndim=2] lin2  = np.ascontiguousarray(-lin )
    cdef np.ndarray[np.double_t, ndim=1] offset_lin2  = np.ascontiguousarray(offset_lin )
    
    cdef double *pt_lin
    cdef double *pt_lin_offset
    if numlin > 0:
        pt_lin_offset = &offset_lin2[0]
        pt_lin = &lin2[0, 0]

    cdef np.ndarray[np.double_t, ndim=1] initial2 = np.ascontiguousarray(initial)

    cdef int seed = np.random.randint(1, 100000)

    cdef double *samples_Carray = <double *>malloc(n_sample*p * sizeof(double))
    
    samples(n_sample, 
            p,
            seed,
            &initial2[0],
            numlin,
            numquad,
            pt_lin,
            pt_quad,
            pt_quad_lin,
            pt_lin_offset,
            pt_quad_offset,
            samples_Carray)


    cdef np.ndarray[np.double_t, ndim=2] samples_array = np.zeros((n_sample, p))
    for i in range(n_sample):
        for j in range(p):
            samples_array[i, j] = samples_Carray[i*p + j]

    free(samples_Carray)

    return samples_array

    
