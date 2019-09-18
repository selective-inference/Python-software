from copy import copy
from functools import wraps
import collections
import numpy as np
import nose
import nose.tools

try:
    from numpy.testing.decorators import SkipTest
except (ImportError, AttributeError):
    from numpy.testing import SkipTest

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
        def set_seed_func(*args, **kwargs):
            """Set_Seed for normal test functions."""
            if set_val():
                old_state = np.random.get_state()
                np.random.seed(seed)
            value = f(*args, **kwargs)
            if set_val():
                np.random.set_state(old_state)
            return value

        @wraps(f)
        def set_seed_gen(*args, **kwargs):
            """Set_Seed for test generators."""
            if set_val():
                old_state = np.random.get_state()
                np.random.seed(seed)
            for x in f(*args, **kwargs):
                yield x
            if set_val():
                np.random.set_state(old_state)

        # Choose the right set_seed to use when building the actual decorator.
        if nose.util.isgenerator(f):
            set_seed = set_seed_gen
        else:
            set_seed = set_seed_func

        return nose.tools.make_decorator(f)(set_seed)

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

def rpy_test_safe(libraries=[], msg=None):
    """
    Loads rpy2 for a test if available, otherwise skips it.

    Parameters
    ----------
    libraries : libraries to load in R

    Returns
    -------
    decorator : function
        Decorator which  when applied to a function, sets the
        random seed before running the test and then
        restores numpy's random state after running the test.
    Notes
    -----
    The decorator itself is decorated with the ``nose.tools.make_decorator``
    function in order to transmit function name, and various other metadata.
    """

    def rpy_safe_decorator(f):

        try:
            import rpy2.robjects as rpy
            from rpy2.robjects import numpy2ri
            
            if libraries:
                for library in libraries:
                    rpy.r('library(%s)' % library)
            rpy_loaded = True    
        except ImportError:
            rpy_loaded = False

        def get_msg(func,msg=None):
            """Skip message with information about function being skipped."""
            if msg is None:
                out = 'Test skipped due to test condition'
            else:
                out = msg

            return "Skipping test: %s: %s" % (func.__name__, out)

        @wraps(f)
        def modified_func(*args, **kwargs):
            """Modified for normal test functions."""
            if not rpy_loaded:
                raise SkipTest(get_msg(f, msg))
            else:
                return f(*args, **kwargs)

        @wraps(f)
        def modified_gen(*args, **kwargs):
            """Modified for test generators."""
            if rpy_loaded:
                kwargs_cp = copy(kwargs)
                kwargs_cp.update(sampling_params)
                for x in f(*args, **kwargs_cp):
                    yield x
            else:
                raise np.testing.decorators.SkipTest(get_msg(f, msg))

        # Choose the right modified to use when building the actual decorator.
        if nose.util.isgenerator(f):
            modified = modified_gen
        else:
            modified = modified_func

        return nose.tools.make_decorator(f)(modified)

    return rpy_safe_decorator
