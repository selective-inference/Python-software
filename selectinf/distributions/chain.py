"""
A simple implementation of Besag and Clifford's generalized
Monte Carlo Significance Tests.

.. _besag_clifford: http://www.jstor.org/stable/pdf/2336623.pdf

"""

import numpy as np

class markov_chain(object):

    """
    Abstract implementation of a Markov chain.

    
    """

    # API

    # A Markov chain knows how to step forward

    def forward_step(self):
        raise NotImplementedError('abstract method')

    # Some Markov chains can run in time-reversed direction.
    # Not all subclasses need this method implemented

    def backward_step(self):
        raise NotImplementedError('abstract method')

    # A Markov chain has a state

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state

    state = property(get_state, set_state)

    # A Markov chain is iterable

    def __iter__(self):
        return self

    def next(self):
        return self.forward_step()

    __next__ = next # Python3 compatibility

class reversible_markov_chain(markov_chain):

    """
    An abstract representation of a Markov chain that
    is reversible with respect to some stationary
    distribution.
    """

    # For reversible chains forward_step
    # and backward_step are the same

    def step(self):
        return NotImplementedError('abstract method')

    def forward_step(self):
        """
        For a reversible chain, a forward_step is just a step.
        """
        return self.step()

    def backward_step(self):
        """
        For a reversible chain, a backward_step is just a step.
        """
        return self.step()

def parallel_test(reversible_chain, null_state, test_statistic, ndraw=20):
    """

    Besag and Clifford's parallel test for reversible
    Markov chains.

    Parameters
    ----------

    reversible_chain : iterable
        An object implementing a Markov chain,
        with `forward_step` and `backward_step` methods.

    null_state : object
        An object nominally drawn from the
        stationary distribution.

    test_statistic : callable
        A test statistic to compute on each state
        of the chain. The overall
        test statistic is the ranking of
        `test_statistic(null_state)` in a sample
        of `ndraw` steps of the chain.
        
    ndraw : int
        How many total draws of the chain should be made?
        Includes `null_state` as one of these draws.

    Returns
    -------

    rank : int
        How many of the draws had a test statistic
        less than the observed value? 
        Ties are handled by randomization.

    Notes
    -----

    The attribute `chain.state` is reset to its initial value
    after running.

    """

    chain = reversible_chain

    observed = test_statistic(null_state)

    results = []

    old_state, chain.state = chain.state, null_state
    
    intermed_state = chain.backward_step()
    
    for _ in range(ndraw-1):

        # take a step from intermediate state
        chain.forward_step()

        # compute the test statistic
        results.append(test_statistic(chain.state))

        # reset the state to the intermediate state
        chain.state = intermed_state

    results = sorted(results)

    rank = np.sum([r < observed for r in results]) 
    ties = np.sum([observed == r for r in results]) 
    
    possible_ranks = range(rank, rank + ties + 1)
    final_rank = np.random.choice(possible_ranks)

    # reset the chain's state to previous value

    chain.state = old_state

    return final_rank

# make sure nose does not try to test this function
parallel_test.__test__ = False

def serial_test(reversible_chain, null_state, test_statistic, ndraw=20):
    """

    Besag and Clifford's parallel test for reversible
    Markov chains.

    Parameters
    ----------

    reversible_chain : iterable
        An object implementing a Markov chain,
        with next method returning current state.

    null_state : object
        An object nominally drawn from the
        stationary distribution.

    test_statistic : callable
        A test statistic to compute on each state
        of the chain. The overall
        test statistic is the ranking of
        `test_statistic(null_state)` in a sample
        of `ndraw` steps of the chain.
        
    ndraw : int
        How many total draws of the chain should be made?
        Includes `null_state` as one of these draws.
        Ties are handled by randomization.

    Returns
    -------

    rank : int
        How many of the draws had a test statistic
        less than the observed value?

    Notes
    -----

    The attribute `chain.state` is reset to its initial value
    after running.

    """

    chain = reversible_chain

    observed = test_statistic(null_state)

    results = []

    old_state, chain.state = chain.state, null_state
    
    random_idx = np.random.random_integers(low=0, high=ndraw-1)

    # go forward from null_state

    for _ in range(random_idx):

        chain.forward_step()

        # compute the test statistic
        results.append(test_statistic(chain.state))

    # reset the state

    chain.state = null_state

    # go backward from null_state

    for _ in range(ndraw - 1 - random_idx):
        chain.backward_step()

        # compute the test statistic
        results.append(test_statistic(chain.state))

    results = sorted(results)

    rank = np.sum([r < observed for r in results]) 
    ties = np.sum([observed == r for r in results]) 

    possible_ranks = range(rank, rank + ties + 1)
    final_rank = np.random.choice(possible_ranks)
    
    # reset the chain's state to previous value

    chain.state = old_state

    return final_rank

# make sure nose does not try to test this function
serial_test.__test__ = False
