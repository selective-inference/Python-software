import tensorflow as tf
import numpy as np

def extract_weights(keys,sess):
    weights_map = {}
    for var in tf.trainable_variables():
        if var.name in keys:
            weights_map[var.name] = sess.run(var)
    return weights_map

def create_l2_loss(output,labels):
    loss = tf.losses.mean_squared_error(labels=labels, predictions=output)
    return loss

def create_optimizer(nsteps=1000,batch_size=20,opt=tf.train.GradientDescentOptimizer(0.01)):	
    def optimize(create_predictor,create_loss,X,Y,weights=None):
        tf.reset_default_graph()
        session = tf.Session()
        x,y = tf.data.Dataset.from_tensor_slices((X,Y)).shuffle(1000).repeat().batch(batch_size).make_one_shot_iterator().get_next()
        output = create_predictor(x)
        loss = create_loss(output,y)
        keys = []
        for k in loss.graph.get_collection('trainable_variables'):
            keys.append(k.name)
        train = opt.minimize(loss)
	
	# initialize weights
        if weights is None:
            init_op = tf.global_variables_initializer()
            session.run(init_op)
        else:
            assign_fn = tf.contrib.framework.assign_from_values_fn(weights)
            assign_fn(session)
		
        # train
        for i in range(nsteps):
            _, loss_val = session.run((train,loss))
            print('optimizing=',i,loss_val)
        return extract_weights(keys,session)
    return optimize

def predict(weights,create_predictor,X):
    # predict on all the data...
    tf.reset_default_graph()
    session = tf.Session()
    x = tf.data.Dataset.from_tensor_slices(X).batch(X.shape[0]).make_one_shot_iterator().get_next()
    predictor = create_predictor(x)
    assign_fn = tf.contrib.framework.assign_from_values_fn(weights)
    assign_fn(session)
    pred = session.run(predictor)
    print('predictions on a all data -- might crash if it is big',pred)
    return pred


def fit(Xtrain,Y,create_predictor,create_loss,optimize):
    print('x0=',Xtrain.shape)
    print('x1=',Y.shape)
    weights = optimize(create_predictor,create_loss,Xtrain,Y)
    def create_pr(Xtest):
        return predict(weights,create_predictor,Xtest)
    return create_pr

