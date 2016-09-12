from copy import copy
import collections
import numpy as np

def set_seed_for_test(seed=10):
    """
    Fix the seed for random test.

    Parameters
    ----------
    seed : int
        Random seed passed to np.random.seed

    Returns
    -------
    decorator : function
        Decorator which, when applied to a function, sets the 
        random seed before running the test and then
        restores numpy's random state after running the test.

    Notes
    -----
    The decorator itself is decorated with the ``nose.tools.make_decorator``
    function in order to transmit function name, and various other metadata.


    """
    import nose
    def set_seed_decorator(f):

        def skipper_func(*args, **kwargs):
            """Skipper for normal test functions."""
            old_state = np.random.get_state()
            np.random.seed(seed)
            value = f(*args, **kwargs)
            np.random.set_state(old_state)
            return value

        def skipper_gen(*args, **kwargs):
            """Skipper for test generators."""
            old_state = np.random.get_state()
            np.random.seed(seed)
            for x in f(*args, **kwargs):
                yield x
            np.random.set_state(old_state)

        # Choose the right skipper to use when building the actual decorator.
        if nose.util.isgenerator(f):
            skipper = skipper_gen
        else:
            skipper = skipper_func

        return nose.tools.make_decorator(f)(skipper)

    return set_seed_decorator

def set_sampling_params_iftrue(condition, nsim=10, burnin=5, ndraw=5):
    """
    Fix the seed for random test.

    Parameters
    ----------

    condition : bool or callable
        Flag to determine whether to set sampling parameters in the decorated test.

    seed : int
        Random seed passed to np.random.seed

    Returns
    -------
    decorator : function
        Decorator which, when applied to a function, sets the 
        random seed before running the test and then
        restores numpy's random state after running the test.

    Notes
    -----
    The decorator itself is decorated with the ``nose.tools.make_decorator``
    function in order to transmit function name, and various other metadata.


    """
    import nose

    def set_params_decorator(f):

        # Allow for both boolean or callable set conditions.
        if isinstance(condition, collections.Callable):
            set_val = lambda : condition()
        else:
            set_val = lambda : condition

        def modified_func(*args, **kwargs):
            """Modified for normal test functions."""
            if set_val():
                kwargs_cp = copy(kwargs)
                kwargs_cp['nsim'] = nsim
                kwargs_cp['burnin'] = burnin
                kwargs_cp['ndraw'] = ndraw
                value = f(*args, **kwargs_cp)
            else:
                value = f(*args, **kwargs)
            return value

        def modified_gen(*args, **kwargs):
            """Modified for test generators."""
            if set_val():
                kwargs_cp = copy(kwargs)
                kwargs_cp['nsim'] = nsim
                kwargs_cp['burnin'] = burnin
                kwargs_cp['ndraw'] = ndraw
                for x in f(*args, **kwargs_cp):
                    yield x

        # Choose the right modified to use when building the actual decorator.
        if nose.util.isgenerator(f):
            modified = modified_gen
        else:
            modified = modified_func

        return nose.tools.make_decorator(f)(modified)

    return set_params_decorator

