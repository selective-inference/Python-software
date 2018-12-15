import numpy as np

from learn_selection.core import cross_inference
from learn_selection.keras_fit import keras_fit

data = np.load('lasso_multi_learning.npz')
learning_data = (data['T'][:2000], data['Y'][:2000])

result = cross_inference(learning_data, 
                         data['nuisance'],
                         data['direction'],
                         keras_fit,
                         fit_args={'epochs':3, 'sizes':[10]*5, 'dropout':0., 'activation':'relu'})
