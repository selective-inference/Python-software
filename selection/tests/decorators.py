from copy import copy
from functools import wraps
import collections
import numpy as np
import nose
import nose.tools

from .reports import reports

def set_seed_iftrue(condition, seed=10):
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

    def set_seed_decorator(f):

        # Allow for both boolean or callable set conditions.
        if isinstance(condition, collections.Callable):
            set_val = lambda : condition()
        else:
            set_val = lambda : condition

        @wraps(f)
        def skipper_func(*args, **kwargs):
            """Skipper for normal test functions."""
            if set_val():
                old_state = np.random.get_state()
                np.random.seed(seed)
            value = f(*args, **kwargs)
            if set_val():
                np.random.set_state(old_state)
            return value

        @wraps(f)
        def skipper_gen(*args, **kwargs):
            """Skipper for test generators."""
            if set_val():
                old_state = np.random.get_state()
                np.random.seed(seed)
            for x in f(*args, **kwargs):
                yield x
            if set_val():
                np.random.set_state(old_state)

        # Choose the right skipper to use when building the actual decorator.
        if nose.util.isgenerator(f):
            skipper = skipper_gen
        else:
            skipper = skipper_func

        return nose.tools.make_decorator(f)(skipper)

    return set_seed_decorator

def set_sampling_params_iftrue(condition, **sampling_params):
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

    def set_params_decorator(f):

        # Allow for both boolean or callable set conditions.
        if isinstance(condition, collections.Callable):
            set_val = lambda : condition()
        else:
            set_val = lambda : condition

        @wraps(f)
        def modified_func(*args, **kwargs):
            """Modified for normal test functions."""
            if set_val():
                kwargs_cp = copy(kwargs)
                kwargs_cp.update(sampling_params)
                value = f(*args, **kwargs_cp)
            else:
                value = f(*args, **kwargs)
            return value

        @wraps(f)
        def modified_gen(*args, **kwargs):
            """Modified for test generators."""
            if set_val():
                kwargs_cp = copy(kwargs)
                kwargs_cp.update(sampling_params)
                for x in f(*args, **kwargs_cp):
                    yield x

        # Choose the right modified to use when building the actual decorator.
        if nose.util.isgenerator(f):
            modified = modified_gen
        else:
            modified = modified_func

        return nose.tools.make_decorator(f)(modified)

    return set_params_decorator

def wait_for_return_value(max_tries=50, strict=True):
    """
    Decorate a test to make it wait until the test
    returns something.
    """

    def wait_for_decorator(test):

        @wraps(test)
        def _new_test(*args, **kwargs):
            count = 0
            while True:
                count += 1
                v = test(*args, **kwargs)
                if v is not None:
                    return count, v
                if count >= max_tries:
                    raise ValueError('test has not returned anything after %d tries' % max_tries)
        return nose.tools.make_decorator(test)(_new_test)

    return wait_for_decorator

def register_report(columns):
    """
    Register a report in selection.tests.reports
    that can be used to create simulation results
    """

    def register_decorator(test):

        @wraps(test)
        def _new_test(*args, **kwargs):
            return test(*args, **kwargs)
        if hasattr(test, 'func_name'): # Py2.*
            name = test.func_name
        else:
            name = test.__name__       # Py3.*
        if name in reports:
            print('Overwriting existing report %s' % name)
        reports[name] = {'test':_new_test, 'columns':columns}
        return nose.tools.make_decorator(test)(_new_test)

    return register_decorator
