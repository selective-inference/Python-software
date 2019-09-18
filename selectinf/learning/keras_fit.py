'''
Based on https://stackoverflow.com/questions/44164749/how-does-keras-handle-multilabel-classification
'''

import warnings

try:

    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import SGD

    def keras_fit(T, Y, **kwargs):

        if Y.ndim == 1:
            Y.shape = (-1, 1)

        fitfns = []

        for j in range(Y.shape[1]):
            y = Y[:,j]

            fit_fn = keras_fit_multilabel(T, y, **kwargs)[0]
            fitfns.append(fit_fn)
        return fitfns

    def keras_fit_multilabel(T, Y, sizes=[500, 500], epochs=50, activation='relu', dropout=0, **ignored):

        if Y.ndim == 1:
            Y.shape = (-1, 1)

        model = Sequential()
        for s in sizes:
            model.add(Dense(s, activation=activation, input_dim=T.shape[1]))
            if dropout > 0:
                model.add(Dropout(dropout))

        # the final layer
        model.add(Dense(Y.shape[1], activation='sigmoid'))

        sgd = SGD(lr=0.03, decay=1e-3, momentum=0.6, nesterov=True)
        model.compile(loss='binary_crossentropy',
                      optimizer=sgd)

        model.fit(T, Y, epochs=epochs)
        fitfns = [lambda T_test: model.predict(T_test)[:,j] for j in range(Y.shape[1])]
        return fitfns

except ImportError:
    warnings.warn('module `keras` not importable, `keras_fit` and `keras_fit_multilabel` will not be importable')
