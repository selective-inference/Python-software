import warnings
import numpy as np, cython
cimport numpy as cnp

DTYPE_float = np.float
ctypedef cnp.float_t DTYPE_float_t
DTYPE_int = np.int

cdef extern from "selective_mle.h":

    double barrier_solve(double *gradient,                   # Gradient vector
                         double *opt_variable,               # Optimization variable
                         double *opt_proposed,               # New value of optimization variable
                         double *conjugate_arg,              # Argument to conjugate of Gaussian
                         double *precision,                  # Precision matrix of Gaussian
                         double *scaling,                    # Diagonal scaling matrix for log barrier
                         int ndim,                           # Dimension of opt_variable
                         int max_iter,                       # Maximum number of iterations
                         int min_iter,                       # Minimum number of iterations
                         double value_tol,                   # Tolerance for convergence based on value
                         double initial_step)                # Initial stepsize

    double barrier_solve_affine(double *gradient,                   # Gradient vector
                                double *opt_variable,               # Optimization variable
                                double *opt_proposed,               # New value of optimization variable
                                double *conjugate_arg,              # Argument to conjugate of Gaussian
                                double *precision,                  # Precision matrix of Gaussian
                                double *scaling,                    # Diagonal scaling matrix for log barrier
                                double *linear_term,                # Matrix A in constraint Au \leq b
                                double *offset,                     # Offset b in constraint Au \leq b
                                double *affine_term,                # Should be equal to b - A.dot(opt_variable)
                                int ndim,                           # Dimension of conjugate_arg, precision
                                int ncon,                           # Number of constraints
                                int max_iter,                       # Maximum number of iterations
                                int min_iter,                       # Minimum number of iterations
                                double value_tol,                   # Tolerance for convergence based on value
                                double initial_step);               # Initial step size

def barrier_solve_(cnp.ndarray[DTYPE_float_t, ndim=1] gradient ,     # Gradient vector
                   cnp.ndarray[DTYPE_float_t, ndim=1] opt_variable,  # Optimization variable
                   cnp.ndarray[DTYPE_float_t, ndim=1] opt_proposed,  # New value of optimization variable
                   cnp.ndarray[DTYPE_float_t, ndim=1] conjugate_arg, # Argument to conjugate of Gaussian
                   cnp.ndarray[DTYPE_float_t, ndim=2] precision,     # Precision matrix of Gaussian
                   cnp.ndarray[DTYPE_float_t, ndim=1] scaling,       # Diagonal scaling matrix for log barrier
                   double initial_step,
                   int max_iter=1000,
                   int min_iter=50,
                   double value_tol=1.e-8):

    ndim = precision.shape[0]

    value = barrier_solve(<double *>gradient.data,
                           <double *>opt_variable.data,
                           <double *>opt_proposed.data,
                           <double *>conjugate_arg.data,
                           <double *>precision.data,
                           <double *>scaling.data,
                           ndim,
                           max_iter,
                           min_iter,
                           value_tol,
                           initial_step)

    barrier_hessian = lambda u, v: (-1./((v + u)**2.) + 1./(u**2.))
    hess = np.linalg.inv(precision + np.diag(barrier_hessian(opt_variable, scaling)))
    return value, opt_variable, hess

def barrier_solve_affine_(cnp.ndarray[DTYPE_float_t, ndim=1] gradient ,     # Gradient vector
                          cnp.ndarray[DTYPE_float_t, ndim=1] opt_variable,  # Optimization variable
                          cnp.ndarray[DTYPE_float_t, ndim=1] opt_proposed,  # New value of optimization variable
                          cnp.ndarray[DTYPE_float_t, ndim=1] conjugate_arg, # Argument to conjugate of Gaussian
                          cnp.ndarray[DTYPE_float_t, ndim=2] precision,     # Precision matrix of Gaussian
                          cnp.ndarray[DTYPE_float_t, ndim=1] scaling,       # Diagonal scaling matrix for log barrier
                          cnp.ndarray[DTYPE_float_t, ndim=2] linear_term,   # Linear part of affine constraint: A
                          cnp.ndarray[DTYPE_float_t, ndim=1] offset,        # Offset part of affine constraint: b
                          cnp.ndarray[DTYPE_float_t, ndim=1] affine_term,   # b - A.dot(opt)
                          double initial_step,
                          int max_iter=1000,
                          int min_iter=50,
                          double value_tol=1.e-8):

    ndim = precision.shape[0]
    ncon = linear_term.shape[0]

    value = barrier_solve_affine(<double *>gradient.data,
                                  <double *>opt_variable.data,
                                  <double *>opt_proposed.data,
                                  <double *>conjugate_arg.data,
                                  <double *>precision.data,
                                  <double *>scaling.data,
                                  <double *>linear_term.data,
                                  <double *>offset.data,
                                  <double *>affine_term.data,
                                  ndim,
                                  ncon,
                                  max_iter,
                                  min_iter,
                                  value_tol,
                                  initial_step)

    final_affine = offset - linear_term.dot(opt_variable)
    barrier_hessian = lambda u, v: (-1./((v + u)**2.) + 1./(u**2.))
    hess = np.linalg.inv(precision + linear_term.T.dot(np.diag(barrier_hessian(final_affine, scaling))).dot(linear_term))
    return value, opt_variable, hess

def solve_barrier_nonneg(conjugate_arg,
                         precision,
                         feasible_point,
                         step=1,
                         max_iter=2000,
                         min_iter=200,
                         tol=1.e-12):

    gradient = np.zeros_like(conjugate_arg)
    opt_variable = np.asarray(feasible_point)
    opt_proposed = opt_variable.copy()
    scaling = 1 / np.sqrt(np.diag(precision))
  
    return barrier_solve_(gradient,
                          opt_variable,
                          opt_proposed,
                          conjugate_arg,
                          precision,
                          scaling,
                          step,
                          max_iter=max_iter,
                          min_iter=min_iter,
                          value_tol=tol)

def solve_barrier_affine(conjugate_arg,
                         precision,
                         feasible_point,
                         linear_term,
                         offset,
                         step=1,
                         max_iter=1000,
                         min_iter=200,
                         tol=1.e-12):

    gradient = np.zeros_like(conjugate_arg)
    opt_variable = np.asarray(feasible_point)
    opt_proposed = opt_variable.copy()
    affine_term = np.zeros_like(offset)
    A = linear_term

    scaling = 1 / np.sqrt(np.diag(A.dot(precision).dot(A.T)))

    linear_term_fortran = np.asfortranarray(linear_term)

    value, opt_variable, hess = barrier_solve_affine_(gradient,
                                                      opt_variable,
                                                      opt_proposed,
                                                      conjugate_arg,
                                                      precision,	
                                                      scaling,
                                                      linear_term_fortran,
                                                      offset,
                                                      affine_term,
                                                      step,
                                                      max_iter=max_iter,
                                                      min_iter=min_iter,
                                                      value_tol=tol
						      )

    return value, opt_variable, hess