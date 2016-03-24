import numpy as np

from selection.algorithms.tests.test_lasso import test_data_carving

P = []
covered = []

num_except = 0
for _ in range(500):
    try:
        results = test_data_carving(compute_intervals=True,
                                    burnin=5000,
                                    ndraw=10000)[0]
        covered.extend(results[-4])
        P.extend(results[0])
        print np.mean(P), np.std(P), 'null'
        print np.mean(covered), 'covered'
        
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        num_except += 1; print('num except: %d' % num_except); pass
        pass
        

