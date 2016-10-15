import numpy as np
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection
#from selection.bayesian.sel_probability import selection_probability
from selection.bayesian.sel_probability2 import cube_subproblem, cube_gradient, cube_barrier, selection_probability_objective
#from selection.bayesian.dual_optimization import dual_selection_probability
from selection.randomized.api import randomization
import regreg.api as rr


#fixing n, p, true sparsity and signal strength
n=20
p=3
s=2
snr=5

#sampling the Gaussian instance
X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=n, p=p, s=s, sigma=1, rho=0, snr=snr)
random_Z = np.random.standard_normal(p)
#getting randomized Lasso solution
lam, epsilon, active, betaE, cube, initial_soln = selection(X,y, random_Z)
lagrange=lam*np.ones(p)

lagrange = lam * np.ones(p)

class dual_test(rr.smooth_atom):

    def __init__(self,
                 X,
                 feasible_point,
                 active,
                 active_signs,
                 lagrange,
                 mean_parameter,
                 noise_variance,
                 randomization,
                 epsilon,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.X=X
        n, p = X.shape
        E = active.sum()

        self.active = active
        self.noise_variance = noise_variance
        self.randomization = randomization

        self.CGF_randomization = randomization.CGF

        if self.CGF_randomization is None:
            raise ValueError(
                'randomization must know its cgf -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = lagrange[~active]

        X_E = self.X_E = X[:,active]
        B = X.T.dot(X_E)

        B_E = B[active]
        B_mE = B[~active]

        self.A_active = np.hstack([(B_E + epsilon * np.identity(E)) * active_signs[None, :],np.zeros((E,p-E))])
        self.A_inactive = np.hstack([B_mE * active_signs[None, :],np.identity((p-E))])
        self.A=np.vstack((self.A_active,self.A_inactive))
        self.dual_arg = np.zeros(p)
        self.dual_arg[:E] = -active_signs * lagrange[active]
        self.feasible_point=feasible_point

        initial=feasible_point

