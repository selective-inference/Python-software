import numpy as np
import regreg.api as rr

from ..cv_glmnet import CV_glmnet
from ...tests.instance import gaussian_instance

def test_cv_glmnet():
    np.random.seed(2)
    n, p = 3000, 1000
    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=30, rho=0., sigma=1)
    loss = rr.glm.gaussian(X,y)
    CV_glmnet_gaussian = CV_glmnet(loss, 'gaussian')
    lam_CV, lam_1SD, lam_seq, CV_err, SD = CV_glmnet_gaussian.using_glmnet()
    print("CV error curve (nonrandomized):", CV_err)
    lam_grid_size = CV_glmnet_gaussian.lam_seq.shape[0]
    lam_CVR, SD, CVR, CV1, lam_seq = CV_glmnet_gaussian.choose_lambda_CVR(scale1=0.1, scale2=0.1)
    print("nonrandomized index:", list(lam_seq).index(lam_CV)) # index of the minimizer
    print("lam for nonrandomized CV plus sigma rule:",lam_CV,lam_1SD)
    print("lam_CVR:",lam_CVR)
    print("randomized index:", list(lam_seq).index(lam_CVR))



