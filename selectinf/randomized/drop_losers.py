from __future__ import print_function, division

import numpy as np
import pandas as pd

from .query import gaussian_query

from .randomization import randomization

class drop_losers(gaussian_query):

    def __init__(self,
                 df,   # should have columns 'arm', 'stage', 'data'
                 K=1): # how many should we move forward?

        self.df = df
        self.K = K

        grouped_arm = df.groupby('arm')
        self.std = grouped_arm.std()['data']
        self.means = grouped_arm.mean()['data']
        self.stages = dict([(k, v) for k, v in df.groupby('stage')])
        stage1 = df['stage'].min()
        stage2 = df['stage'].max()
        
        df1 = self.stages[stage1]
        df2 = self.stages[stage2]

        stage1_means = df1.groupby('arm').mean().sort_values('data', ascending=False)
        self._winners = sorted(list(stage1_means.index[:K]))
        best_loser = stage1_means['data'].iloc[K]

        n1 = df1.groupby('arm').count()
        n2 = df2.groupby('arm').count()
        self._n1_win = n1_win = np.array([n1.loc[lambda df: df.index == winner]['data'].iloc[0] 
                                          for winner in self._winners])
        self._n2_win = n2_win = np.array([n2.loc[lambda df: df.index == winner]['data'].iloc[0] 
                                          for winner in self._winners])
        std_win = self.std.loc[self._winners]

        A = -np.identity(K)
        b = -np.ones(K) * best_loser
        linear = np.identity(K)
        offset = np.zeros(K)
        
        # Work out the implied randomization variance
        # Let X1=X[stage1].mean(), X2=X[stage2].mean() and Xf = X.mean()
        # with n1=len(stage1), n2=len(stage2)

        # X1 = Xf + n2/n1 * (Xf-X2)
        #    = Xf + n2/(n1+n2) * (X1-X2)
        # so randomization term is w=n2/(n1+n2) * (X1-X2)
        # with variance 
        # n2**2 / (n1+n2)**2 * (1/n1 + 1/n2) 
        # = n2**2 / (n1+n2)**2 * (n1+n2) / (n1*n2)
        # = n2 / (n1 * (n1 + n2))

        mult = n2_win / (n1_win * (n1_win + n2_win))

        # needed for gaussian_query api

        self.randomizer = randomization.gaussian(np.diag(std_win**2) * mult)
        self.observed_opt_state = stage1_means['data'].iloc[:K]
        self.observed_score_state = -self.means[self._winners] # problem is a minimization
        self.selection_variable = {'winners':self._winners}

        self._setup_sampler(A, b, linear, offset)

    def MLE_inference(self,
                      level=0.9,
                      solve_args={'tol':1.e-12}):
        """

        Parameters
        ----------

        level : float, optional
            Confidence level.

        solve_args : dict, optional
            Arguments passed to solver.

        """
        
        observed_target = self.means[self._winners]
        std_win = self.std.loc[self._winners]
        target_cov = np.diag(std_win**2 / (self._n1_win + self._n2_win))
        target_score_cov = -target_cov
        
        result = gaussian_query.selective_MLE(self,
                                              observed_target,
                                              target_cov,
                                              target_score_cov,
                                              level=level,
                                              solve_args=solve_args)
        result[0].insert(0, 'arm', self._winners)
        return result

    def summary(self,
                level=0.9,
                ndraw=10000,
                burnin=2000):

        """
        Produce p-values and confidence intervals for targets
        of model including selected features

        Parameters
        ----------

        level : float
            Confidence level.

        ndraw : int (optional)
            Defaults to 1000.

        burnin : int (optional)
            Defaults to 1000.

        """
        observed_target = self.means[self._winners]
        std_win = self.std.loc[self._winners]
        target_cov = np.diag(std_win**2 / (self._n1_win + self._n2_win))
        target_score_cov = -target_cov

        result = gaussian_query.summary(self,
                                        observed_target,
                                        target_cov,
                                        target_score_cov,
                                        alternatives=['twosided']*self.K,
                                        ndraw=ndraw,
                                        level=level,
                                        burnin=burnin,
                                        compute_intervals=True)
        result.insert(0, 'arm', self._winners)
        return result
                             
