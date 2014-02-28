import numpy as np

from .constraints import constraints
from .truncated import truncated_gaussian, UMAU_interval, naive_interval

class SPRT(object):

    DEBUG = False
    alpha = 0.05

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

    def __call__(self, Z, sigma=1):
        """
        Perform the sequential test.

        Parameters
        ----------

        Z : `np.float`
           The observed pairs of data.

        Returns
        -------

        strp_results : `STRP_results`
           Container for results from `STRP`.

        """

        Z = np.asarray(Z)
        diff = Z[:,1] - Z[:,0]
        stopped = False
        stopping_time, running_sum = 0, 0
        boundary = []

        outcome = 'reached max time'
        while True:
            running_sum += diff[stopping_time]
            stopping_time += 1
            upper_bound = sigma * self.upper_boundary(stopping_time)
            lower_bound = sigma * self.lower_boundary(stopping_time)

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

        # Form the constraints

        A = np.tril(np.ones((n,n)))[(nmin-1):-1]

        # If the boundary is hit, there should be
        # two intervals, each formed from 2(n-nmin)+1 constraints

        diff = diff[:n]

        if stopped:

            last_row = np.ones(n)
            m = A.shape[0]
            AU = np.vstack([A,-A,last_row.reshape((1,-1))])
            bU = np.zeros(2*m+1)
            bU[:m] = -boundary[(nmin-1):-1,0]
            bU[m:2*m] = boundary[(nmin-1):-1,1]
            bU[-1] = -upper_bound

            conU = constraints((AU,bU), None)
            conU.covariance *= sigma**2

            eta = np.ones(n)

            L1, _, U1, V = conU.bounds(eta, diff)

            AL = AU.copy()
            AL[-1] *= -1
            bL = bU.copy()
            bL[-1] = lower_bound
            conL = constraints((AL,bL), None)
            conL.covariance *= sigma**2

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
            con.covariance *= sigma**2

            eta = np.ones(n)
            L1, _, U1, V = con.bounds(eta, diff)

            intervals = ((L1, U1))
            observed = (eta*diff).sum()
            selection_constraints = [con]

        tg = truncated_gaussian(intervals, sigma=np.sqrt(V))
        tg.use_R = False
        naive = naive_interval(observed, 
                                        self.alpha, 
                                        tg)
        naive = np.array(naive) / n
        
        umau_interval = UMAU_interval(observed,
                                      self.alpha, 
                                      tg)
        umau_interval = np.array(umau_interval) / n

        return SPRT_result(stopping_time, 
                           naive,
                           umau_interval, 
                           boundary, 
                           selection_constraints, 
                           tg, 
                           outcome, 
                           observed / n)

class SPRT_result(object):

    def __init__(self, stopping_time, 
                 naive_interval, 
                 umau_interval, 
                 boundary,
                 selection_constraints,
                 trunc_gauss,
                 outcome,
                 observed):

        self.stopping_time = stopping_time
        self.naive_interval = naive_interval
        self.umau_interval = umau_interval
        self.boundary = boundary
        self.selection_constraints = selection_constraints
        self.trunc_gauss = trunc_gauss
        self.outcome = outcome
        self.observed = observed
