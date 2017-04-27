# Do it like tutourial
# For running the model
import lasagne
import theano
from theano import tensor as T
import numpy as np

# For loading the dataset:
import os
import gzip
import urllib


def build_model(optimizing_strategy):
    ## CONFIGURATION ##
    # configuration for first layer
    conv_1_filters = 6
    conv_1_size = 5
    pool_1_size = 2
    # configuration for second layer
    conv_2_filters = 10
    conv_2_size = 5
    pool_2_size = 2
    # number of units for fc1
    num_units_fc1 = 100
    # We will run the example in mnist - 10 digits
    output_size=10
    # Define optimization strategy
    #optimizing_strategy = 'Gradientenabstieg'
    #optimizing_strategy = 'RMSProp'
    #optimizing_strategy = 'AdaDelta'
    ###################

    # Batch size x Img Channels x Height x Width
    data_size=(None,1,28,28)

    # Symbolic variables
    # Tensor for input Data-Set
    input_var = T.tensor4('input')
    # Vector for description of correct digit
    target_var = T.ivector('targets')

    net = {}

    # Input layer:
    net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

    # Convolution + Pooling
    # conv_1_filters count of filters, conv_1_size size of filter
    net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=conv_1_filters, filter_size=conv_1_size)
    net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=pool_1_size)

    # conv_2_filters count of filters, conv_2_size size of filter
    net['conv2'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=conv_2_filters, filter_size=conv_2_size)
    net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=pool_2_size)

    # Fully-connected + dropout
    net['fc1'] = lasagne.layers.DenseLayer(net['pool2'], num_units=num_units_fc1)
    net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'],  p=0.5)

    # Output layer:
    net['out'] = lasagne.layers.DenseLayer(net['drop1'], num_units=output_size, nonlinearity=lasagne.nonlinearities.softmax)

    # Define hyperparameters. These could also be symbolic variables
    lr = 1e-2
    weight_decay = 1e-5

    # Loss function: squared_error
    prediction = lasagne.layers.get_output(net['out'])
    # A vector of losses
    loss = lasagne.objectives.squared_error(prediction, target_var)
    # Average of the losses in MINI-BATCH
    loss = loss.mean()

    # Also add weight decay to the cost function
    # Add regularization to cost fuction
    weightsl2 = lasagne.regularization.regularize_network_params(net['out'], lasagne.regularization.l2)
    # Update loss funktion with calculated regulatization
    loss += weight_decay * weightsl2

    #Get the update rule for Stochastic Gradient Descent with Nesterov Momentum
    params = lasagne.layers.get_all_params(net['out'], trainable=True)
    updates = 0

    if optimizing_strategy == 'Gradientenabstieg':
        updates = lasagne.updates.sgd(loss, params, learning_rate=lr)

    elif optimizing_strategy == 'RMSProp':
        updates = lasagne.updates.sgd(loss, params, learning_rate=lr)

    elif optimizing_strategy == 'AdaDelta':
        updates = lasagne.updates.sgd(loss, params, learning_rate=lr)

    else:
        print('Error while executing optimization')

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    test_prediction = lasagne.layers.get_output(net['out'], deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    get_preds = theano.function([input_var], test_prediction)

    return (train_fn, val_fn, get_preds)

dataset_file = 'mnist.pkl.gz'

#Download dataset if not yet done:
if not os.path.isfile(dataset_file):
    urllib.urlretrieve('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz', dataset_file)

#Load the dataset
f = gzip.open(dataset_file, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#Convert the dataset to the shape we want
x_train, y_train = train_set
x_test, y_test = test_set

x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

import time
epochs = 10
batch_size=100

#Run the training function per mini-batches
n_examples = x_train.shape[0]
n_batches = n_examples / batch_size

def train(train_fn):
    cost_history = []
    for epoch in range(epochs):
        st = time.time()
        batch_cost_history = []
        for batch in range(n_batches):
            x_batch = x_train[batch*batch_size: (batch+1) * batch_size]
            y_batch = y_train[batch*batch_size: (batch+1) * batch_size]

            this_cost = train_fn(x_batch, y_batch) # This is where the model gets updated

            batch_cost_history.append(this_cost)
        epoch_cost = np.mean(batch_cost_history)
        cost_history.append(epoch_cost)
        en = time.time()
        print('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds' % (epoch+1, epochs, epoch_cost, en-st))
    return cost_history

sgd_functions = build_model('Gradientenabstieg')
rmsprop_functions = build_model('RMSProp')
adam_functions = build_model('AdaDelta')

print ("Training with Gradientenabstieg")
sgd_cost_history = train(sgd_functions[0])
print ("Training with RMSProp")
rmsprop_cost_history = train(rmsprop_functions[0])
print ("Training with AdaDelta")
adam_cost_history = train(adam_functions[0])

#Plot the cost history

import matplotlib.pyplot as plt
#%matplotlib inline

plt.figure(figsize=(7,5))
x = range(1,11)
plt.plot(x, sgd_cost_history, 'b-^')
plt.plot(x, rmsprop_cost_history, 'g-*')
plt.plot(x, adam_cost_history, 'r-d')

plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.legend(['SGD','RMSProp','ADAM'])
plt.savefig('training_loss.png')

def get_error(val_fn):
    loss, acc = val_fn(x_test, y_test)
    test_error = 1 - acc
    return test_error

print("Model trained with SGD. Test error: %f" % get_error(sgd_functions[1]))
print("Model trained with RMSProp. Test error: %f" % get_error(rmsprop_functions[1]))
print("Model trained with ADAM. Test error: %f" % get_error(adam_functions[1]))
