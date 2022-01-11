import numpy as np

def target_query_Interactspec(query_spec,
                              regress_target_score,
                              cov_target):

    QS = query_spec
    prec_target = np.linalg.inv(cov_target)

    U1 = regress_target_score.T.dot(prec_target)
    U2 = U1.T.dot(QS.M2.dot(U1))
    U3 = U1.T.dot(QS.M3.dot(U1))
    U5 = U1.T.dot(QS.M4)
    U4 = QS.M4.dot(QS.cond_cov).dot(U5.T)

    return U1, U2, U3, U4, U5



