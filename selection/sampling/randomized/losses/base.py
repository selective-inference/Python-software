import numpy as np
import regreg.api as rr

class selective_loss(rr.smooth_atom):

    ### begin API

    ### selective loss API

    def gradient(self, data, beta):
        """
        Gradient of smooth part.
        """
        raise NotImplementedError("abstract method")

    def hessian(self, data, beta):
        """
        Hessian of smooth part.
        """
        raise NotImplementedError("abstract method")

    def log_jacobian(self, data, beta):
        """
        Log-Jacobian of smooth part.
        Active subspace should have columns that
        """
        raise NotImplementedError("abstract method")

    def setup_sampling(self, 
                       y, 
                       quadratic_coef, 
                       *args):
        raise NotImplementedError("abstract method")

    def proposal(self, data):
        """
        Metropolis-Hastings proposal to move `data`.
        """
        raise NotImplementedError("abstract method")

    def logpdf(self, y):
        """
        logpdf of `data`, refers to density `f` in the manuscript.
        """
        raise NotImplementedError("abstract method")

    def update_proposal(self, y, proposal, log_ratio):
        """
        Update state of loss based on current data,
        proposal and the acceptance probability of the step
        from y to proposal.
        """
        raise NotImplementedError("abstract method")

    def step_data(self, state, logpdf, val): #ADDED VAL

        self.total_data += 1

        data, opt_vars = state

        proposal, log_transition_ratio = self.proposal(data, val) ## ADDED VAL

        #return proposal

        proposal_state = (proposal, opt_vars)

        log_ratio = (log_transition_ratio
                     + logpdf(proposal_state)
                     - logpdf(state))

        self.update_proposal(data, proposal, log_ratio)

        if np.log(np.random.uniform()) < log_ratio:
            self.accept_data += 1
            data = proposal

        return data
