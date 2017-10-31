import warnings
import numpy as np, cython
cimport numpy as np

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

cdef extern from "debias.h":

    void multiply_by_2(double *X, int nval)

def foo(np.ndarray[DTYPE_float_t, ndim=1] A):
    multiply_by_2(<double *>A.data, A.shape[0]) 
    print('here')
    return A

#    int solve_wide(double *X_ptr,              # Sqrt of non-neg def matrix -- X^TX/ncase = nndef #
#                   double *X_theta_ptr,        # Fitted values   #
#                   double *linear_func_ptr,    # Linear term in objective #
#                   double *nndef_diag_ptr,     # Diagonal entries of non-neg def matrix #
#                   double *gradient_ptr,       # X^TX/ncase times theta + linear_func#
#                   int *need_update_ptr,       # Keeps track of updated gradient coords #
#                   int *ever_active_ptr,       # Ever active set: 1-based # 
#                   int *nactive_ptr,           # Size of ever active set #
#                   int ncase,                  # How many rows in X #
#                   int nfeature,               # How many columns in X #
#                   double *bound_ptr,          # Lagrange multipliers #
#                   double ridge_term,          # Ridge / ENet term #
#                   double *theta_ptr,          # current value #
#                   double *theta_old_ptr,      # previous value #
#                   int maxiter,                # max number of iterations #
#                   double kkt_tol,             # precision for checking KKT conditions #
#                   double objective_tol,       # precision for checking relative decrease in objective value #
#                   double parameter_tol,       # precision for checking relative convergence of parameter #
#                   int max_active,             # Upper limit for size of active set -- otherwise break # 
#                   int objective_stop,         # Break based on convergence of objective value? #
#                   int kkt_stop,               # Break based on KKT? #
#                   int param_stop)             # Break based on parameter convergence? #
