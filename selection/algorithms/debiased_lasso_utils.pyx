import warnings
import numpy as np, cython
cimport numpy as np

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

cdef extern from "debias.h":

   int solve_wide(double *X_ptr,              # Sqrt of non-neg def matrix -- X^TX/ncase = nndef #
                  double *X_theta_ptr,        # Fitted values   #
                  double *linear_func_ptr,    # Linear term in objective #
                  double *nndef_diag_ptr,     # Diagonal entries of non-neg def matrix #
                  double *gradient_ptr,       # X^TX/ncase times theta + linear_func#
                  int *need_update_ptr,       # Keeps track of updated gradient coords #
                  int *ever_active_ptr,       # Ever active set: 1-based # 
                  int *nactive_ptr,           # Size of ever active set #
                  int ncase,                  # How many rows in X #
                  int nfeature,               # How many columns in X #
                  double *bound_ptr,          # Lagrange multipliers #
                  double ridge_term,          # Ridge / ENet term #
                  double *theta_ptr,          # current value #
                  double *theta_old_ptr,      # previous value #
                  int maxiter,                # max number of iterations #
                  double kkt_tol,             # precision for checking KKT conditions #
                  double objective_tol,       # precision for checking relative decrease in objective value #
                  double parameter_tol,       # precision for checking relative convergence of parameter #
                  int max_active,             # Upper limit for size of active set -- otherwise break # 
                  int kkt_stop,               # Break based on KKT? #
                  int objective_stop,         # Break based on convergence of objective value? #
                  int parameter_stop)         # Break based on parameter convergence? #

   int check_KKT_wide(double *theta_ptr,        # current theta #
                      double *gradient_ptr,     # X^TX/ncase times theta + linear_func#
                      double *X_theta_ptr,      # Current fitted values #
                      double *X_ptr,            # Sqrt of non-neg def matrix -- X^TX/ncase = nndef #
                      double *linear_func_ptr,  # Linear term in objective #   
                      int *need_update_ptr,     # Which coordinates need to be updated? #
                      int nfeature,             # how many columns in X #
                      int ncase,                # how many rows in X #
                      double *bound_ptr,        # Lagrange multiplers for \ell_1 #
                      double ridge_term,        # Ridge / ENet term #
                      double tol)               # precision for checking KKT conditions #        
   
   void update_gradient_wide(double *gradient_ptr,     # X^TX/ncase times theta + linear_func #
                             double *X_theta_ptr,      # Current fitted values #
                             double *X_ptr,            # Sqrt of non-neg def matrix -- X^TX/ncase = nndef #
                             double *linear_func_ptr,  # Linear term in objective #   
                             int *need_update_ptr,     # Which coordinates need to be updated? #
                             int nfeature,             # how many columns in X #
                             int ncase)                # how many rows in X #
   
def solve_wide_(np.ndarray[DTYPE_float_t, ndim=2] X,            # Sqrt of non-neg def matrix -- X^TX/ncase = nndef 
                np.ndarray[DTYPE_float_t, ndim=1] X_theta,      # Fitted values   #
                np.ndarray[DTYPE_float_t, ndim=1] linear_func,  # Linear term in objective #
                np.ndarray[DTYPE_float_t, ndim=1] nndef_diag,   # Diagonal entries of non-neg def matrix #
                np.ndarray[DTYPE_float_t, ndim=1] gradient,     # X^TX/ncase times theta + linear_func#
                np.ndarray[DTYPE_int_t, ndim=1] need_update,    # Keeps track of updated gradient coords #
                np.ndarray[DTYPE_int_t, ndim=1] ever_active,    # Ever active set: 1-based # 
                np.ndarray[DTYPE_int_t, ndim=1] nactive,        # Size of ever active set #
                np.ndarray[DTYPE_float_t, ndim=1] bound,        # Lagrange multipliers #
                double ridge_term,                              # Ridge / ENet term #
                np.ndarray[DTYPE_float_t, ndim=1] theta,        # current value #
                np.ndarray[DTYPE_float_t, ndim=1] theta_old,    # previous value #
                int maxiter,                                    # max number of iterations #
                double kkt_tol,                                 # precision for checking KKT conditions #
                double objective_tol,                           # precision for checking relative 
                                                                #   decrease in objective value #
                double parameter_tol,                           # precision for checking 
                                                                #   relative convergence of parameter #
                int max_active,                                 # Upper limit for size of active set #
                int kkt_stop,                                   # Break based on KKT? #
                int objective_stop,                             # Break based on convergence of objective value? #
                int parameter_stop):                            # Break based on parameter convergence? #

    niter = solve_wide(<double *>X.data,
                        <double *>X_theta.data,
                        <double *>linear_func.data,
                        <double *>nndef_diag.data,
                        <double *>gradient.data,
                        <int *>need_update.data,
                        <int *>ever_active.data,
                        <int *>nactive.data,
                        <int>X.shape[0],
                        <int>X.shape[1],
                        <double *>bound.data,
                        ridge_term,
                        <double *>theta.data,
                        <double *>theta_old.data,
                        maxiter,
                        kkt_tol,
                        parameter_tol,
                        objective_tol,
                        max_active,
                        kkt_stop,
                        parameter_stop,
                        objective_stop)

    # Check whether feasible

    ncase = X.shape[0]
    nfeature = X.shape[1]

    kkt_check = check_KKT_wide(<double *>theta.data,
                               <double *>gradient.data,
                               <double *>X_theta.data,
                               <double *>X.data,
                               <double *>linear_func.data,
                               <int *>need_update.data,
                               ncase,
                               nfeature,
                               <double *>bound.data,
                               ridge_term,
                               kkt_tol)

    max_active_check = nactive[0] >= max_active

    # Make sure gradient is updated -- essentially a matrix multiply

    update_gradient_wide(<double *>gradient.data,
                          <double *>X_theta.data,
                          <double *>X.data,
                          <double *>linear_func.data,
                          <int *>need_update.data,
                          ncase,
                          nfeature)

    return {'soln':theta,
            'gradient':gradient,
            'X_theta':X_theta,
            'linear_func':linear_func,
            'iter':niter,
            'kkt_check':kkt_check,
            'ever_active':ever_active,
            'nactive':nactive,
            'max_active_check':max_active_check}
              
