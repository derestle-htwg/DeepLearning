import time
import numpy as np
import theano
import theano.tensor as T
from sklearn import cross_validation
import lasagne
import pandas as pd
from neupy.datasets import make_reber_classification
import matplotlib.pyplot as plt
import csv

#b = open('reber_classification_examples.csv', 'w')
#a = csv.writer(b)
#Words, labels = make_reber_classification(5000, invalid_size=0.5)
#X_test = make_reber_classification(400, invalid_size=0.5)
#data = [[Words],
#        [labels],
#        [X_test]]
#a.writerows(data)
#b.close()

print("Generating Reber Grammar Words classification dataset")
wordSize = 7
Words, labels = make_reber_classification(4000, invalid_size=0.5, wordSize=wordSize)
X_test = make_reber_classification(500, invalid_size=0.5, wordSize=wordSize)

def getAlphabetWithAllWords(arr):

    X_str = ""
    # Develop string over all words
    for i in range(len(arr)):
        X_str = X_str + arr[i]

    # Whole string
    # print(X_str)

    # Convert to list
    Alphabet = list(set(X_str))
    global Alphabet_enc
    #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
    Alphabet_enc = pd.get_dummies(pd.DataFrame(Alphabet))

    # Add description for columns
    Alphabet_enc.columns = Alphabet

    # return whole matix with all dummy rows
    return Alphabet_enc


def getEncodeWord(word):
    # Empty Erray
    Word_enc = []

    # Iterate through all words and create arrays with binary idendity-encoding
    for i in range(len(word)):
        Word_enc.append(np.asarray(Alphabet_enc[word[i]]).tolist()[0:])

    # Concat the [0,0,1,0,0] Arrays for thw whole words
    Word_enc = sum(Word_enc, [])

    # Concat all words to one string
    return np.array(Word_enc)


def OH_Encoding(X_set, y_set):
    # Make string with all examples
    Alphabet = getAlphabetWithAllWords(X_set)

    # There are 5 different characters
    # Longest word has 26 characters
    # Make shape (5000, 130, 1)
    X_enc = np.zeros(   shape=( len(X_set),
                        max([len(s) for s in X_set]) * len(Alphabet),
                        1))

    # Make shape (5000, 130)
    mask = np.zeros(    shape=( len(X_set),
                        max([len(s) for s in X_set]) * len(Alphabet)))

    # Make shape (5000, 2) with TRUE and FALSE opportunity
    y_enc = np.zeros(   shape=[ y_set.shape[0],
                                2])

    # Time measurement
    # start_time = time.time()

    # For every example
    for i in range(len(X_set)):

        # Encode all words with binary-idendity coding to one bis array.
        Encoded_Word = getEncodeWord(X_set[i])
        #print(Encoded_Word)

        #
        X_enc[i, 0:len(Encoded_Word), 0] = Encoded_Word
        #print(X_enc)

        # ceate mask
        for j in range(len(Encoded_Word)):
            mask[i, j] = 1.

        # write encoding for correct labels
        if y_set[i] == 1.:
            y_enc[i] = [1., 0.]
        else:
            y_enc[i] = [0., 1.]
    #print(y_enc)
    #print(mask)
    #print(X_enc)

    # print('Time to encode data : {:.03f}s'.format(time.time() - start_time))
    print("Data completely encoded")
    return X_enc, y_enc, mask


def Split_Data(X_set, y_set, val_size):
    # Encoding of reber input-data for approp. matrix-shapes
    X_enc, y_enc, mask = OH_Encoding(X_set, y_set)

    # Divide whole data-set into train and validation-data
    X_train, X_val, y_train, y_val = cross_validation.train_test_split(X_enc, y_enc, test_size=val_size, random_state=1260)
    # Do same for mask
    mask_train, mask_val = cross_validation.train_test_split(mask, test_size=val_size, random_state=1260)

    # Return tratin and validation datasets
    return X_train, X_val, y_train, y_val, mask_train, mask_val

# Create Batch with specific Size
def batch_gen(X, y, Mask, N):
    while True:
        # Choose a example at random
        idx = np.random.choice(len(y), N)
        # Create the three stuctures (Input (x), label (y), )
        yield X[idx].astype('float32'), y[idx].astype('float32'), Mask[idx].astype('float32')


def LSTM(MAX_LENGTH, N_HIDDEN1):
    # Start building a LSTM Network

    # First a inputLayser with Shape X,X,1 Last is for true or false.
    l_in = lasagne.layers.InputLayer(shape=(None, None, 1))

    # Input-Layer Max_length = size of first dimension of mask-train-set
    l_mask = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH))

    # Initialize recurrent layers (gate and cell parameter)
    gate_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(), b=lasagne.init.Constant(0.))
    cell_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(), W_cell=None, b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.sigmoid)

    # First LSTM lasagne layer
    # forward
    l_lstm1 = lasagne.layers.recurrent.LSTMLayer(l_in, N_HIDDEN1, mask_input=l_mask, ingate=gate_parameters, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, learn_init=True, grad_clipping=100.)
    # backward
    l_lstm1_back = lasagne.layers.recurrent.LSTMLayer(l_in, N_HIDDEN1, ingate=gate_parameters, mask_input=l_mask, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, learn_init=True, grad_clipping=100., backwards=True)
    # sum
    l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm1, l_lstm1_back])

    ### Second LSTM Lasagne Layer
    #l_lstm2 = lasagne.layers.recurrent.LSTMLayer(l_sum, N_HIDDEN1, mask_input=l_mask, ingate=gate_parameters, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, learn_init=True, grad_clipping=100.)
    #l_lstm2_back = lasagne.layers.recurrent.LSTMLayer(l_sum, N_HIDDEN1, ingate=gate_parameters, mask_input=l_mask, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, learn_init=True, grad_clipping=100., backwards=True)
    #l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm2, l_lstm2_back])

    # Slices the input at a specific axis and at specific indices. - (incoming feeding layer, indices, axis=-1, **kwargs)
    l_lstm_slice = lasagne.layers.SliceLayer(l_sum, 0, 1)

    # A fully connected layer
    l_out = lasagne.layers.DenseLayer(l_lstm_slice, num_units=2, nonlinearity=lasagne.nonlinearities.sigmoid)

    # return a triple with functions
    return l_in, l_mask, l_out


def Train_model(BATCH_SIZE, number_of_epochs, lr, backprobOption):

    # Create valid datasets for LSTM-Network - Separate them into train, valid and mask dataset.
    # val_size describes Mini-Batch size
    X_train, X_val, y_train, y_val, mask_train, mask_val = Split_Data(Words, labels, val_size=0.3)

    # Lstm Network configuration
    l_in, l_mask, l_out = LSTM(MAX_LENGTH=mask_train.shape[1], N_HIDDEN1=128)

    # Theano tensor variables
    y_sym = T.matrix()
    mask = T.matrix()

    # get outputs
    output = lasagne.layers.get_output(l_out)

    # if output greather then 50 percent
    pred = (output > 0.5)

    # Normalisation
    loss = T.mean(lasagne.objectives.binary_crossentropy(output, y_sym))
    acc = T.mean(T.eq(pred, y_sym))

    params = lasagne.layers.get_all_params(l_out)
    grad = T.grad(loss, params)

    # Choose the optimization strategie
    updates = []
    if(backprobOption == 'momentum'):
        updates = lasagne.updates.momentum(grad, params, learning_rate=lr)
    elif(backprobOption == 'adadelta'):
        updates = lasagne.updates.adadelta(grad, params)
    elif(backprobOption == 'rmsprop'):
        updates = lasagne.updates.rmsprop(grad, params)
    else:
        print('Error while choosing backprobOption')

    # Get functions for training, validation and prediction
    f_train = theano.function([l_in.input_var, y_sym, l_mask.input_var], [loss, acc], updates=updates)
    f_val = theano.function([l_in.input_var, y_sym, l_mask.input_var], [loss, acc])
    f_predict = theano.function([l_in.input_var, l_mask.input_var], pred)

    # Get batches for training, validation
    # // means division without digits after the decimal point
    N_BATCHES = len(X_train) // BATCH_SIZE
    N_VAL_BATCHES = len(X_val) // BATCH_SIZE
    train_batches = batch_gen(X_train, y_train, mask_train, BATCH_SIZE)
    val_batches = batch_gen(X_val, y_val, mask_val, BATCH_SIZE)

    # Store learning state after each epoch into a list
    global predictions
    predictions = []

    lossArray = []
    print("Start training")
    for epoch in range(number_of_epochs):
        # Stop time
        start_time = time.time()

        # Training with all examples in Batches
        for _ in range(N_BATCHES):
            X, y, mask = next(train_batches)
            loss, acc = f_train(X, y, mask)
            predictions.append(f_predict(X, mask))

        # Proof with validation examples
        val_loss = 0
        val_acc = 0
        for _ in range(N_VAL_BATCHES):
            X, y, mask = next(val_batches)
            loss, acc = f_val(X, y, mask)
            val_loss += loss
            val_acc += acc
        # Calculate total percentage of
        val_loss /= N_VAL_BATCHES
        val_acc /= N_VAL_BATCHES

        # To get all missclassified examples we have to subtract the total accurancy from 1
        lossArray.append(1 - val_acc)

        # Print results per epoch
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, number_of_epochs, time.time() - start_time))
        print('Validation Loss: {:.03f}'.format(1 - val_acc))

    return lossArray


if __name__ == "__main__":

    # Variables
    numberOfEpochs = 20
#
    ## Do different optimizations
    print("Adadelta optimization")
    adadelta_loss = Train_model(BATCH_SIZE=32, number_of_epochs=numberOfEpochs, lr=0.001, backprobOption='adadelta')
    #print(adadelta_loss)
#
    print("Momentum optimization")
    momentum_loss = Train_model(BATCH_SIZE=32, number_of_epochs=numberOfEpochs, lr=0.001, backprobOption='momentum')
    #print(momentum_loss)
#
    print("RMS Propagation optimization")
    rmsprop_loss = Train_model(BATCH_SIZE=32, number_of_epochs=numberOfEpochs, lr=0.001, backprobOption='rmsprop')
    #print(rmsprop_loss)
#
    ## evenly sampled time at 200ms intervals
    #t = np.arange(0, numberOfEpochs, 1)

    # Fake data
    #adadelta_loss = [0.894567, 0.835567, 0.79448372, 0.642221, 0.714493930, 0.559938389]
    #momentum_loss = [0.894567, 0.835567, 0.79448372, 0.642221, 0.714493930, 0.559938389]
    #rmsprop_loss = [0.894567, 0.835567, 0.79448372, 0.642221, 0.714493930, 0.559938389]

    #print the results
    # red dashes, blue squares and green triangles
    t = np.arange(1., numberOfEpochs+1, 1.)
    plt.figure(figsize=(7, 5))
    plt.plot(t, adadelta_loss, 'b-^')
    plt.plot(t, momentum_loss, 'g-*')
    plt.plot(t, rmsprop_loss, 'r-d')
    plt.title('LSTM Network - 1 Layer - Sigmoid - 20 epochs - lr=0.001')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.legend(['adadelta', 'momentum', 'rmsprop'])
    plt.savefig('training_loss.png')
    plt.show()
