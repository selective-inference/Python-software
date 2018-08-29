import tensorflow_fit
import tensorflow as tf
import numpy as np
ntries, sigma, q = 21, 1, 0.3
Z = np.linspace(-8, 8, 1001)
def algorithm(Z, ntries=ntries, q=q):
    proportion = 0
    for _ in range(ntries):
        proportion += ((Z + sigma * np.random.standard_normal() > 0) * 
                       (Z + 1 + sigma * np.random.standard_normal() > 0) *
                       (Z - 0.5 + sigma * np.random.standard_normal() > 0))
    proportion /= ntries
    return proportion > q

Z = np.linspace(-8, 8, 1001)


# a function that is parameterized by hyperparameters
def create_network(num_hidden,num_outputs):
	def create(features):
		N = features.shape[0]
		X = features # np.reshape(features,(None,1))
		hidA = tf.layers.Dense(activation=tf.nn.relu,units=num_hidden, name='hidA')
		outlayer = tf.layers.Dense(activation=tf.nn.relu,units=num_outputs, name='hid')
		#outlayer = tf.layers.Dense(activation=tf.nn.relu, name='hid')
		output = outlayer(hidA(X))
		return output
	return create

def fit_algorithm(algorithm, B=5000, ntries=ntries, q=q, Zval=Z, link='probit'):
    
    Z = np.random.standard_normal(B) * 2
    Z = np.hstack([Z, 
                   np.random.standard_normal(B), 
                   np.random.standard_normal(B) * 3, 
                   np.random.standard_normal(B) * 0.5])
    print('ZS=',Z.shape)
    Y = np.array([algorithm(z, ntries=ntries, q=q) for z in Z])
    optimize = tensorflow_fit.create_optimizer() # a default optimizer
    predictor_f = tensorflow_fit.fit(np.reshape(Z,(Z.shape[0],1)),np.reshape(Y,(Y.shape[0],1)),create_network(10,1),tensorflow_fit.create_l2_loss,optimize) 
    print('ZS2=',Zval.shape)
    return predictor_f(np.reshape(Zval,(Zval.shape[0],1)))

def simulate(ntries=ntries, sigma=sigma, truth=0):
               
    while True:
        Z = np.random.standard_normal() + truth
        if algorithm(Z, ntries, q=q):
            return Z

Z = np.linspace(-8, 8, 1001)
W1 = fit_algorithm(algorithm, ntries=ntries, q=q, Zval=Z)
print('done')
plt.plot(Z, np.log(W1))
selective_law1 = discrete_family(Z, W1 * scipy.stats.norm.pdf(Z))


def pivot1(z, truth=0):
    return 1 - selective_law1.cdf(truth, z)

