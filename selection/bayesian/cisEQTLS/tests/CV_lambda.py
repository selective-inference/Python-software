from rpy2.robjects.packages import importr
from rpy2 import robjects
glmnet = importr('glmnet')
from selection.tests.instance import gaussian_instance
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import numpy as np
import regreg.api as rr
from selection.api import randomization
from selection.bayesian.initial_soln import selection, instance


def tuning_parameter_glmnet(X, y):
    robjects.r('''
        glmnet_cv = function(X,y, lam_seq=NA){
        y = as.matrix(y)
        X = as.matrix(X)
        if (is.na(lam_seq)){
            G_CV = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE)
        }
        else {
            G_CV = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE, lambda=lam_seq)
        }
        lam_1SE = G_CV$lambda.1se
        lam_minCV = G_CV$lambda.min
        n = nrow(X)
        lam_minCV = lam_minCV*n
        lam_1SE = lam_1SE*n
        lam_seq = G_CV$lambda*n
        result = list(lam_minCV=lam_minCV, lam_1SE=lam_1SE, lam_seq = lam_seq, CV_err=G_CV$cvm, SD=G_CV$cvsd)
        return(result)
        }''')

    r_glmnet_cv = robjects.globalenv['glmnet_cv']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    result = r_glmnet_cv(r_X, r_y)
    lam_minCV = result[0][0]
    lam_1SE = result[1][0]
    return lam_minCV, lam_1SE

sample = instance(n=350, p=5000, s=10, sigma=1, rho=0, snr=5.)

X, y, true_beta, nonzero, noise_variance = sample.generate_response()
lam_CV, lam_1SD = tuning_parameter_glmnet(X, y)
print("tuning parameters", lam_CV)