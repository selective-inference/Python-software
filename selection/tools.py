import time
from functools import wraps


dict_time = dict()


def timethis(func): 
    '''
    Decorator that reports the execution time.
    '''
    dict_time[func.__name__] = (0, 0)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs) 
        end = time.time() 
        #print(func.__name__, end-start) 
        
        k, t = dict_time[func.__name__]
        dict_time[func.__name__] = k+1, t + end-start

        return result
    return wrapper

