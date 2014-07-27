import kmeans
import numpy as np
kmeans = reload(kmeans)

n_sample = 100
p_array = []
for i in range(n_sample):
    if i%10 == 0:
        print i, " / ", n_sample
        
    kmeans = reload(kmeans)
    p = kmeans.f(10)
    p_array.append(p)



import matplotlib.pyplot as plt

p_array = sorted(p_array)

x = np.arange(0, 1, 1./len(p_array));
plt.plot(x, p_array, 'ro')
