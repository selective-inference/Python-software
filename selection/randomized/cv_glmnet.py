from rpy2.robjects.packages import importr
from rpy2 import robjects
glmnet = importr('glmnet')
from selection.tests.instance import gaussian_instance
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import numpy as np
import regreg.api as rr
from selection.api import randomization

class CV_glmnet(object):

    def __init__(self, loss, loss_label):
        self.loss = loss
        if loss_label=="gaussian":
            self.family=robjects.StrVector('g')
        elif loss_label=="logistic":
            self.family=robjects.StrVector('b')

    def using_glmnet(self, loss=None):
        robjects.r('''
            glmnet_cv = function(X,y, family, lam_seq=NA){
            y = as.matrix(y)
            X = as.matrix(X)
            if (family=="b"){
                family.full = "binomial"
            } else if (family=="g"){
                family.full = "gaussian"
            }
            if (is.na(lam_seq)){
                G_CV = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE, family=family.full)
            }
            else {
                G_CV = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE, lambda=lam_seq, family=family.full)

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

        if loss is None:
            loss = self.loss
        X, y = loss.data
        r_glmnet_cv = robjects.globalenv['glmnet_cv']
        n, p = X.shape
        r_X = robjects.r.matrix(X, nrow=n, ncol=p)
        r_y = robjects.r.matrix(y, nrow=n, ncol=1)
        if not hasattr(self, 'lam_seq'):
            result = r_glmnet_cv(r_X, r_y, self.family)
        else:
            r_lam_seq = robjects.r.matrix(np.true_divide(self.lam_seq, n), nrow=self.lam_seq.shape[0], ncol=1)
            result = r_glmnet_cv(r_X, r_y, self.family, r_lam_seq)
        lam_minCV = result[0][0]
        lam_1SE = result[1][0]
        lam_seq = np.array(result[2])
        if not hasattr(self, 'lam_seq'):
            self.lam_seq = lam_seq
        CV_err = np.array(result[3])
        # this is stupid but glmnet sometime cuts my given seq of lambdas
        if CV_err.shape[0]<self.lam_seq.shape[0]:
            CV_err_longer = np.ones(self.lam_seq.shape[0])*np.max(CV_err)
            CV_err_longer[:(self.lam_seq.shape[0]-1)]=CV_err
            CV_err = CV_err_longer
        SD = np.array(result[4])
        #print("lam_minCV", lam_minCV)
        return lam_minCV, lam_1SE, lam_seq, CV_err, SD


    def choose_lambda_CVR(self,  scale1 = None, scale2=None, loss=None):
        if loss is None:
            loss = self.loss
        _, _, _, CV_err, SD = self.using_glmnet(loss)

        rv1, rv2 = np.zeros(self.lam_seq.shape[0]), np.zeros(self.lam_seq.shape[0])
        if scale1 is not None:
            randomization1 = randomization.isotropic_gaussian((self.lam_seq.shape[0],), scale=scale1)
            rv1 = np.asarray(randomization1._sampler(size=(1,)))
        if scale2 is not None:
            randomization2 = randomization.isotropic_gaussian((self.lam_seq.shape[0],), scale=scale2)
            rv2 = np.asarray(randomization2._sampler(size=(1,)))
        CVR = CV_err+rv1.flatten()+rv2.flatten()
        lam_CVR = self.lam_seq[np.argmin(CVR)] # lam_CVR minimizes CVR
        #print("randomized index:", list(self.lam_seq).index(lam_CVR))
        CV1 = CV_err+rv1.flatten()
        return  lam_CVR, SD, CVR, CV1, self.lam_seq


    def bootstrap_CVR_curve(self, scale1=None, scale2=None):
        """
        Bootstrap of CVR=CV+R1+R2 and CV1=CV+R1 curves
        """

        def _bootstrap_CVerr_curve(indices):
            loss_star = self.loss.subsample(indices)
            # loss_star = rr.glm.gaussian(X[indices,:], y[indices])
            _, _, CVR_val, CV1_val, _ = self.choose_lambda_CVR(scale1, scale2, loss_star)
            return np.array(CVR_val), np.array(CV1_val)

        def _CVR_boot(indices):
            return _bootstrap_CVerr_curve(indices)[0]

        def _CV1_boot(indices):
            return _bootstrap_CVerr_curve(indices)[1]

        return _CVR_boot, _CV1_boot


if __name__ == '__main__':
    np.random.seed(2)
    n, p = 3000, 1000
    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=30, rho=0., sigma=1)
    loss = rr.glm.gaussian(X,y)
    CV_glmnet_compute = CV_glmnet(loss)
    lam_CV, lam_1SD, lam_seq, CV_err, SD = CV_glmnet_compute.using_glmnet()
    print("CV error curve (nonrandomized):", CV_err)
    lam_grid_size = CV_glmnet_compute.lam_seq.shape[0]
    lam_CVR, SD, CVR, CV1, lam_seq = CV_glmnet_compute.choose_lambda_CVR(scale1=0.1, scale2=0.1)
    print("nonrandomized index:", list(lam_seq).index(lam_CV)) # index of the minimizer
    print("lam for nonrandomized CV plus sigma rule:",lam_CV,lam_1SD)
    print("lam_CVR:",lam_CVR)
    print("randomized index:", list(lam_seq).index(lam_CVR))
    import matplotlib.pyplot as plt
    plt.plot(np.log(lam_seq), CV_err)
    plt.plot(np.log(lam_seq), CVR)
    #plt.ylabel('some numbers')
    plt.show()


