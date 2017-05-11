#For running the model:
import lasagne
import theano
from theano import tensor as T
import numpy as np

#For loading the dataset:
import os
import gzip
import cPickle
import urllib

data_size = (None, 1, 28, 28)  # Batch size x Img Channels x Height x Width
output_size = 10  # We will run the example in mnist - 10 digits


def build_model(optimizer):
    """Builds the model, loss function and optimization functions.
    The param optimizer sould be in ["sgd", "adam", "rmsprop"]  """

    input_var = T.tensor4('input')
    target_var = T.imatrix('targets')
    net = {}

    # Input layer:
    net['data'] = lasagne.layers.InputLayer(data_size, input_var=input_var)

    # Convolution + Pooling
    net['conv1'] = lasagne.layers.Conv2DLayer(net['data'], num_filters=6, filter_size=5)
    net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'], pool_size=2)

    net['conv2'] = lasagne.layers.Conv2DLayer(net['pool1'], num_filters=10, filter_size=5)
    net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'], pool_size=2)

    # Fully-connected + dropout
    net['fc1'] = lasagne.layers.DenseLayer(net['pool2'], num_units=100)
    net['drop1'] = lasagne.layers.DropoutLayer(net['fc1'], p=0.5)

    # Output layer:
    net['out'] = lasagne.layers.DenseLayer(net['drop1'], num_units=output_size,
                                           nonlinearity=lasagne.nonlinearities.softmax)

    weight_decay = 1e-5

    # Loss function: mean cross-entropy
    prediction = lasagne.layers.get_output(net['out'])
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()

    # Also add weight decay to the cost function
    weightsl2 = lasagne.regularization.regularize_network_params(net['out'], lasagne.regularization.l2)
    loss += weight_decay * weightsl2

    # Get the update rule
    params = lasagne.layers.get_all_params(net['out'], trainable=True)
    if (optimizer == 'sgd'):
        updates = lasagne.updates.sgd(
            loss, params, learning_rate=1e-2)
    elif (optimizer == 'adam'):
        updates = lasagne.updates.adam(loss, params)
    elif (optimizer == 'rmsprop'):
        updates = lasagne.updates.rmsprop(loss, params, learning_rate=1e-2)

    test_prediction = lasagne.layers.get_output(net['out'], deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates, name='train')
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], name='validation')
    get_preds = theano.function([input_var], test_prediction, name='get_preds')

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
y_train = [ y_train == 0,
            y_train == 1,
            y_train == 2,
            y_train == 3,
            y_train == 4,
            y_train == 5,
            y_train == 6,
            y_train == 7,
            y_train == 8,
            y_train == 9]
y_test = y_test.astype(np.int32)
y_test = [ y_test == 0,
           y_test == 1,
           y_test == 2,
           y_test == 3,
           y_test == 4,
           y_test == 5,
           y_test == 6,
           y_test == 7,
           y_test == 8,
           y_test == 9]
y_train = y_train * 1
y_test = y_test * 1

y_train = np.matrix(y_train).T
y_test = np.matrix(y_test).T

import time
epochs = 10
batch_size=100

#Run the training function per mini-batches
n_examples = x_train.shape[0]
n_batches = n_examples / batch_size

def train(train_fn):
    cost_history = []
    for epoch in xrange(epochs):
        st = time.time()
        batch_cost_history = []
        for batch in xrange(n_batches):
            x_batch = x_train[batch*batch_size: (batch+1) * batch_size]
            y_batch = y_train[batch*batch_size: (batch+1) * batch_size]

            this_cost = train_fn(x_batch, y_batch) # This is where the model gets updated

            batch_cost_history.append(this_cost)
        epoch_cost = np.mean(batch_cost_history)
        cost_history.append(epoch_cost)
        en = time.time()
        print('Epoch %d/%d, train error: %f. Elapsed time: %.2f seconds' % (epoch+1, epochs, epoch_cost, en-st))
    return cost_history

sgd_functions = build_model('sgd')
rmsprop_functions = build_model('rmsprop')
adam_functions = build_model('adam')

import matplotlib.pyplot as plt





print ("Training with SGD")
#sgd_cost_history = train(sgd_functions[0])
print ("Training with RMSPROP")
#rmsprop_cost_history = train(rmsprop_functions[0])
print ("Training with ADAM")
adam_cost_history = train(adam_functions[0])




#plt.figure(figsize=(7,5))
#x = range(1,11)
#plt.plot(x, sgd_cost_history, 'b-^')
#plt.plot(x, rmsprop_cost_history, 'g-*')
#plt.plot(x, adam_cost_history, 'r-d')

#plt.xlabel('Epoch')
#plt.ylabel('Training loss')
#plt.legend(['SGD','RMSProp','ADAM'])
#plt.savefig('training_loss.png')

#def get_error(val_fn):
 #   loss, acc = val_fn(x_test, y_test)
  #  test_error = 1 - acc
   # return test_error

#print "Model trained with SGD. Test error: %f" % get_error(sgd_functions[1])
#print "Model trained with RMSProp. Test error: %f" % get_error(rmsprop_functions[1])
#print "Model trained with ADAM. Test error: %f" % get_error(adam_functions[1])

plt.show()

for i in range(50000):
    res = adam_functions[2]([x_test[i]])
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(x_test[i][0])
    plt.subplot(212)
    plt.plot([0,1,2,3,4,5,6,7,8,9],res[0], "ro")
    plt.show()
