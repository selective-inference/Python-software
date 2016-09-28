import numpy as np
from selection.algorithms.change_point import one_jump_instance, change_point

def test_change_point(delta=0.1, p=60, sigma=1, plot=False):

    y, signal = one_jump_instance(delta, p, sigma)
    CP = change_point(y)
    fit, relaxed_fit, summary, segments = CP.fit()
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.scatter(np.arange(y.shape[0]), y)
        plt.plot(fit, 'r', label='Penalized', linewidth=3)
        plt.plot(relaxed_fit, 'k', label='Relaxed', linewidth=3)
        plt.plot(signal, 'g', label='Truth', linewidth=3)
        plt.legend(loc='upper left')
    return segments
