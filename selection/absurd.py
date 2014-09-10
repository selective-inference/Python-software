import kmeans
import numpy as np

kmeans = reload(kmeans)

n = 20
p = 5
n_sample = 50
p_array = []

t_distance = [0]
#distance = 5

import matplotlib.pyplot as plt
x = np.arange(0, 1, 1./n_sample);
plt.plot(x, x, 'g')

for distance in t_distance:
    i=0
    while i < n_sample:
        compteur_bug = 0
        if True: #i%1 == 0:
            print i, " / ", n_sample, distance
        try:
            #kmeans = reload(kmeans)
            p_value = kmeans.f(n, p, distance)[0]
            if p_value > 0 and p_value < 1:
                p_array.append(p_value)
                i+=1
        except:
            raise
    

    

    p_array = sorted(p_array)
    print p_array
    
    plt.plot(x, p_array, 'b')



plt.show()
