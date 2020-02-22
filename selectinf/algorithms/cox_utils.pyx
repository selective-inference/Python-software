import warnings
import numpy as np, cython
cimport numpy as cnp

DTYPE_float = np.float
ctypedef cnp.float_t DTYPE_float_t
DTYPE_int = np.int
ctypedef cnp.int_t DTYPE_int_t
ctypedef cnp.intp_t DTYPE_intp_t

cdef extern from "cox_fns.h":

    void _update_cox_exp(double *linear_pred_ptr, # Linear term in objective 
                         double *exp_ptr,         # stores exp(eta) 
                         double *exp_accum_ptr,   # inner accumulation vector 
                         double *case_weight_ptr, # case weights 
                         long *censoring_ptr,     # censoring indicator 
                         long *ordering_ptr,      # 0-based ordering of times 
                         long *rankmin_ptr,       # 0-based ranking with min tie breaking 
                         long ncase               # how many subjects / times 
                         );       

    void _update_cox_expZ(double *linear_pred_ptr,  # Linear term in objective 
                          double *right_vector_ptr, # Linear term in objective 
                          double *exp_ptr,          # stores exp(eta) 
                          double *expZ_accum_ptr,   # inner accumulation vector 
                          double *case_weight_ptr,  # case weights 
                          long *censoring_ptr,      # censoring indicator 
                          long *ordering_ptr,       # 0-based ordering of times 
                          long *rankmin_ptr,        # 0-based ranking with min tie breaking 
                          long ncase                # how many subjects / times 
                          );       

    void _update_outer_1st(double *linear_pred_ptr,     # Linear term in objective 
                           double *exp_accum_ptr,       # inner accumulation vector 
                           double *outer_accum_1st_ptr, # outer accumulation vector 
                           double *case_weight_ptr,     # case weights 
                           long *censoring_ptr,         # censoring indicator 
                           long *ordering_ptr,          # 0-based ordering of times 
                           long *rankmin_ptr,           # 0-based ranking with min tie breaking 
                           long ncase                   # how many subjects / times 
                           );       

    void _update_outer_2nd(double *linear_pred_ptr,     # Linear term in objective 
                           double *exp_accum_ptr,       # inner accumulation vector  Ze^{\eta} 
                           double *expZ_accum_ptr,      # inner accumulation vector e^{\eta} 
                           double *outer_accum_2nd_ptr, # outer accumulation vector 
                           double *case_weight_ptr,     # case weights 
                           long *censoring_ptr,         # censoring indicator 
                           long *ordering_ptr,          # 0-based ordering of times 
                           long *rankmin_ptr,           # 0-based ranking with min tie breaking 
                           long ncase                   # how many subjects / times 
                           );

    double _cox_objective(double *linear_pred_ptr,     # Linear term in objective 
                          double *inner_accum_ptr,     # inner accumulation vector 
                          double *outer_accum_1st_ptr, # outer accumulation vector 
                          double *case_weight_ptr,     # case weights 
                          long *censoring_ptr,         # censoring indicator 
                          long *ordering_ptr,          # 0-based ordering of times 
                          long *rankmin_ptr,           # 0-based ranking with min tie breaking 
                          long *rankmax_ptr,           # 0-based ranking with max tie breaking 
                          long ncase                   # how many subjects / times 
                          );       

    void _cox_gradient(double *gradient_ptr,        # Where gradient is stored 
                       double *exp_ptr,             # stores exp(eta) 
                       double *outer_accum_1st_ptr, # outer accumulation vector 
                       double *case_weight_ptr,     # case weights 
                       long *censoring_ptr,         # censoring indicator 
                       long *ordering_ptr,          # 0-based ordering of times 
                       long *rankmin_ptr,           # 0-based ranking with min tie breaking 
                       long *rankmax_ptr,           # 0-based ranking with max tie breaking 
                       long ncase                   # how many subjects / times 
                       );

    void _cox_hessian(double *hessian_ptr,          # Where hessian is stored 
                      double *exp_ptr,              # stores exp(eta) 
                      double *right_vector_ptr,     # Right vector in Hessian
                      double *outer_accum_1st_ptr,  # outer accumulation vector used in outer prod "mean"
                      double *outer_accum_2nd_ptr,  # outer accumulation vector used in "2nd" moment
                      double *case_weight_ptr,     # case weights 
                      long *censoring_ptr,          # censoring indicator 
                      long *ordering_ptr,           # 0-based ordering of times 
                      long *rankmax_ptr,            # 0-based ranking with max tie breaking 
                      long ncase                    # how many subjects / times 
                      );
   
def cox_objective(cnp.ndarray[DTYPE_float_t, ndim=1] linear_pred,
                  cnp.ndarray[DTYPE_float_t, ndim=1] exp_buffer,
                  cnp.ndarray[DTYPE_float_t, ndim=1] exp_accum,
                  cnp.ndarray[DTYPE_float_t, ndim=1] outer_1st_accum,
                  cnp.ndarray[DTYPE_float_t, ndim=1] case_weight,
                  cnp.ndarray[DTYPE_int_t, ndim=1] censoring,
                  cnp.ndarray[DTYPE_int_t, ndim=1] ordering,
                  cnp.ndarray[DTYPE_int_t, ndim=1] rankmin,
                  cnp.ndarray[DTYPE_int_t, ndim=1] rankmax,
                  long ncase):

    _update_cox_exp(<double *>linear_pred.data,
                    <double *>exp_buffer.data,
                    <double *>exp_accum.data,
                    <double *>case_weight.data,
                    <long *>censoring.data,
                    <long *>ordering.data,
                    <long *>rankmin.data,
                    ncase)

    _update_outer_1st(<double *>linear_pred.data,
                      <double *>exp_accum.data,
                      <double *>outer_1st_accum.data,
                      <double *>case_weight.data,
                      <long *>censoring.data,
                      <long *>ordering.data,
                      <long *>rankmin.data,
                      ncase)

    return _cox_objective(<double *>linear_pred.data,
                          <double *>exp_accum.data,
                          <double *>outer_1st_accum.data,
                          <double *>case_weight.data,
                          <long *>censoring.data,
                          <long *>ordering.data,
                          <long *>rankmin.data,
                          <long *>rankmax.data,
                          ncase)

def cox_gradient(cnp.ndarray[DTYPE_float_t, ndim=1] gradient,
                 cnp.ndarray[DTYPE_float_t, ndim=1] linear_pred,
                 cnp.ndarray[DTYPE_float_t, ndim=1] exp_buffer,
                 cnp.ndarray[DTYPE_float_t, ndim=1] exp_accum,
                 cnp.ndarray[DTYPE_float_t, ndim=1] outer_1st_accum,
                 cnp.ndarray[DTYPE_float_t, ndim=1] case_weight,
                 cnp.ndarray[DTYPE_int_t, ndim=1] censoring,
                 cnp.ndarray[DTYPE_int_t, ndim=1] ordering,
                 cnp.ndarray[DTYPE_int_t, ndim=1] rankmin,
                 cnp.ndarray[DTYPE_int_t, ndim=1] rankmax,
                 long ncase):
    """
    Compute Cox partial likelihood gradient in place.
    """

    # this computes e^{\eta} and stores cumsum at rankmin

    _update_cox_exp(<double *>linear_pred.data,
                    <double *>exp_buffer.data,
                    <double *>exp_accum.data,
                    <double *>case_weight.data,
                    <long *>censoring.data,
                    <long *>ordering.data,
                    <long *>rankmin.data,
                    ncase)

    _update_outer_1st(<double *>linear_pred.data,
                      <double *>exp_accum.data,
                      <double *>outer_1st_accum.data,
                      <double *>case_weight.data,
                      <long *>censoring.data,
                      <long *>ordering.data,
                      <long *>rankmin.data,
                      ncase)

    _cox_gradient(<double *>gradient.data,
                  <double *>exp_buffer.data,
                  <double *>outer_1st_accum.data,
                  <double *>case_weight.data,
                  <long *>censoring.data,
                  <long *>ordering.data,
                  <long *>rankmin.data,
                  <long *>rankmax.data,
                  ncase)
    
    return gradient

def cox_hessian(cnp.ndarray[DTYPE_float_t, ndim=1] hessian,
                cnp.ndarray[DTYPE_float_t, ndim=1] linear_pred,
                cnp.ndarray[DTYPE_float_t, ndim=1] right_vector,
                cnp.ndarray[DTYPE_float_t, ndim=1] exp_buffer,
                cnp.ndarray[DTYPE_float_t, ndim=1] exp_accum,
                cnp.ndarray[DTYPE_float_t, ndim=1] expZ_accum,
                cnp.ndarray[DTYPE_float_t, ndim=1] outer_1st_accum,
                cnp.ndarray[DTYPE_float_t, ndim=1] outer_2nd_accum,
                cnp.ndarray[DTYPE_float_t, ndim=1] case_weight,
                cnp.ndarray[DTYPE_int_t, ndim=1] censoring,
                cnp.ndarray[DTYPE_int_t, ndim=1] ordering,
                cnp.ndarray[DTYPE_int_t, ndim=1] rankmin,
                cnp.ndarray[DTYPE_int_t, ndim=1] rankmax,
                long ncase):
    """
    Compute Cox partial likelihood gradient in place.
    """

    # this computes e^{\eta} and stores cumsum at rankmin, stored in outer_accum_1st

    _update_cox_exp(<double *>linear_pred.data,
                    <double *>exp_buffer.data,
                    <double *>exp_accum.data,
                    <double *>case_weight.data,
                    <long *>censoring.data,
                    <long *>ordering.data,
                    <long *>rankmin.data,
                    ncase)

    _update_outer_1st(<double *>linear_pred.data,
                      <double *>exp_accum.data,
                      <double *>outer_1st_accum.data,
                      <double *>case_weight.data,
                      <long *>censoring.data,
                      <long *>ordering.data,
                      <long *>rankmin.data,
                      ncase)

    _update_cox_expZ(<double *>linear_pred.data,
                     <double *>right_vector.data,
                     <double *>exp_buffer.data,
                     <double *>expZ_accum.data,
                     <double *>case_weight.data,
                     <long *>censoring.data,
                     <long *>ordering.data,
                     <long *>rankmin.data,
                     ncase)

    _update_outer_2nd(<double *>linear_pred.data,
                      <double *>exp_accum.data,
                      <double *>expZ_accum.data,
                      <double *>outer_2nd_accum.data,
                      <double *>case_weight.data,
                      <long *>censoring.data,
                      <long *>ordering.data,
                      <long *>rankmin.data,
                      ncase)

    _cox_hessian(<double *>hessian.data,
                 <double *>exp_buffer.data,
                 <double *>right_vector.data,
                 <double *>outer_1st_accum.data,
                 <double *>outer_2nd_accum.data,
                 <double *>case_weight.data,
                 <long *>censoring.data,
                 <long *>ordering.data,
                 <long *>rankmax.data,
                 ncase)
    
    return hessian
              
