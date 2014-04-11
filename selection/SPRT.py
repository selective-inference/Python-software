import numpy as np

from .constraints import constraints
from .truncated import truncated_gaussian

class SPRT(object):

    DEBUG = False
    use_R = False

    def __init__(self, upper_boundary, 
                 lower_boundary=None, nmin=5, nmax=20):
        """
        An arbitrary boundary sequential test.

        Parameters
        ----------

        Z : `np.float`
           The observed data.

        upper_boundary : `callable`
           Callable that takes `n` as a parameter to determine 
           upper bound of stopping rule.
           By convention we assume `upper_boundary(0)=0`.

        lower_boundary : `callable`
           Callable that takes `n` as a parameter to determine stopping rule.
           By convention we assume `lower_boundary(0)=0`. 
           Defaults to `lambda n: -upper_boundary(n)`

        nmin : `int`
           Minimum number of samples taken.

        nmax : `int`
           Maximum number of samples taken.

        """
        self.upper_boundary = upper_boundary
        self.lower_boundary = lower_boundary or (lambda n: -upper_boundary(n))

        self.nmin = nmin
        self.nmax = nmax

    def __call__(self, Z, sigma=1, extra_samples=0):
        """
        Perform the sequential test.

        Parameters
        ----------

        Z : `np.float`
           The observed pairs of data.

        sigma : `(float, float)` or `float`
           Variance of each entry of Z.
           If not a tuple or list, assumed to be a float.

        Returns
        -------

        strp_results : `STRP_results`
           Container for results from `STRP`.

        """

        if type(sigma) not in [type(()),type([])]:
            sigma = (float(sigma), float(sigma))
        noise_sd = np.sqrt(sigma[0]**2 + sigma[1]**2)

        Z = np.asarray(Z)
        diff = Z[:,1] - Z[:,0]
        stopped = False
        stopping_time, running_sum = 0, 0
        boundary = []

        outcome = 'reached max time'

        while True:
            running_sum += diff[stopping_time]
            stopping_time += 1
            upper_bound = noise_sd * self.upper_boundary(stopping_time)
            lower_bound = noise_sd * self.lower_boundary(stopping_time)

            boundary.append((lower_bound, upper_bound))

            if ((running_sum > upper_bound or 
                 running_sum < lower_bound) 
                and stopping_time >= self.nmin):
                if running_sum > upper_bound:
                    outcome = 'upper boundary'
                else:
                    outcome = 'lower boundary'
                stopped = True
                break
            if stopping_time >= self.nmax:
                break

        boundary = np.array(boundary)

        n, nmin, nmax = stopping_time, self.nmin, self.nmax # shorthand

        # take some extra steps after stopping

        ntot = min(n+extra_samples, self.nmax)

        # Form the constraints

        A = np.tril(np.ones((n,ntot)))[(nmin-1):-1]
        A[:, n:] = 0

        # If the boundary is hit, there should be
        # two intervals, each formed from 2(n-nmin)+1 constraints

        diff = diff[:ntot]

        if stopped:

            last_row = np.ones(ntot)
            last_row[n:] = 0.
            m = A.shape[0]
            AU = np.vstack([A,-A,last_row.reshape((1,-1))])
            bU = np.zeros(2*m+1)
            bU[:m] = -boundary[(nmin-1):-1,0]
            bU[m:2*m] = boundary[(nmin-1):-1,1]
            bU[-1] = -upper_bound

            conU = constraints((AU,bU), None)
            conU.covariance *= noise_sd**2

            eta = np.ones(ntot)

            L1, _, U1, V = conU.bounds(eta, diff)

            AL = AU.copy()
            AL[-1] *= -1
            bL = bU.copy()
            bL[-1] = lower_bound
            conL = constraints((AL,bL), None)
            conL.covariance *= noise_sd**2

            L2, _, U2, _ = conL.bounds(eta, diff)

            intervals = ((L1, U1), (L2, U2))
            observed = (eta*diff).sum()

            selection_constraints = [conU, conL]

        else:
            m = A.shape[0]
            A = np.vstack([-A,A])
            b = np.zeros(2*m)
            b[:m] = -boundary[(nmin-1):-1,0]
            b[m:2*m] = boundary[(nmin-1):-1,1]

            con = constraints((A,b), None)
            con.covariance *= noise_sd**2

            eta = np.ones(ntot)
            L1, _, U1, V = con.bounds(eta, diff)

            intervals = ((L1, U1))
            observed = (eta*diff).sum()
            selection_constraints = [con]

        tg = truncated_gaussian(np.array(intervals) / ntot, sigma=V / ntot)
        tg.use_R = self.use_R
        return SPRT_result(stopping_time, 
                           ntot, 
                           boundary, 
                           selection_constraints, 
                           tg, 
                           outcome, 
                           observed / ntot)

class SPRT_result(object):

    alpha = 0.05

    def __init__(self, stopping_time, 
                 ntot,
                 boundary,
                 selection_constraints,
                 trunc_gauss,
                 outcome,
                 observed,
                 true_difference=0):

        self.stopping_time = stopping_time
        self.ntot = ntot
        self.boundary = boundary
        self.selection_constraints = selection_constraints
        self.trunc_gauss = trunc_gauss
        self.outcome = outcome
        self.observed = observed
        self.true_difference = true_difference

    @property
    def naive_interval(self):
        if not hasattr(self, "_naive_interval"):
            self._naive_interval = self.trunc_gauss.naive_interval(self.observed, self.alpha)
        return self._naive_interval

    @property
    def nominal_interval(self):
        if not hasattr(self, "_nominal_interval"):
            from selection.truncated import _qnorm
            center, sd = self.observed, self.trunc_gauss.sigma
            q = np.fabs(_qnorm(self.alpha / 2., use_R=True))
            self._nominal_interval = np.array([center-q*sd, center+q*sd])
        return self._nominal_interval

    @property
    def UMAU_interval(self):
        if not hasattr(self, "_UMAU_interval"):
            self._UMAU_interval = self.trunc_gauss.UMAU_interval(self.observed, self.alpha)
        return self._UMAU_interval

    def pvalue(self, truth=0):
        old_mu = self.trunc_gauss.mu
        self.trunc_gauss.mu = truth
        P = self.trunc_gauss.CDF(self.observed)
        pval = min(P, 1-P)
        self.trunc_gauss.mu = old_mu
        return pval
