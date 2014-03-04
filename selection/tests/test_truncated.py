import numpy as np
from selection.truncated import truncated_gaussian

intervals = [(-np.inf,-4),(3,np.inf)]

tg = truncated_gaussian(intervals)

X = np.linspace(-5,5,101)
F = [tg.CDF(x) for x in X]
