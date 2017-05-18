import time
import numpy as np
import theano
import theano.tensor as T
from sklearn import cross_validation
import lasagne
import pandas as pd
from neupy.datasets import make_reber_classification

print("Generating Reber Grammar Words classification dataset")
Words, labels = make_reber_classification(5000, invalid_size=0.5)
X_test = make_reber_classification(400, invalid_size=0.5)


# To count character occurences X.count("char")

def Get_Alphabet(arr):
    # Function to compute the uniques letters contained in a set of words
    #   Return : One Hot encoded uniques letters
    X_str = ""
    for i in range(len(arr)):
        X_str = X_str + arr[i]
    Alphabet = list(set(X_str))
    global Alphabet_enc
    Alphabet_enc = pd.get_dummies(pd.DataFrame(Alphabet))
    Alphabet_enc.columns = Alphabet
    return Alphabet_enc


def Encode_Char(char):
    return np.asarray(Alphabet_enc[char]).tolist()


def Encode_Word(word):
    Word_enc = []
    for i in range(len(word)):
        Word_enc.append(Encode_Char(word[i])[0:])
    Word_enc = sum(Word_enc, [])
    return np.array(Word_enc)


def OH_Encoding(X_set, y_set):
    Alphabet = Get_Alphabet(X_set)
    X_enc = np.zeros(shape=(len(X_set), max([len(s) for s in X_set]) * len(Alphabet), 1))
    mask = np.zeros(shape=(len(X_set), max([len(s) for s in X_set]) * len(Alphabet)))
    y_enc = np.zeros(shape=[y_set.shape[0], 2])
    start_time = time.time()
    for i in range(len(X_set)):
        Encoded_Word = Encode_Word(X_set[i])
        X_enc[i, 0:len(Encoded_Word), 0] = Encoded_Word
        for j in range(len(Encoded_Word)):
            mask[i, j] = 1.
        if y_set[i] == 1.:
            y_enc[i] = [1., 0.]
        else:
            y_enc[i] = [0., 1.]
    print('Time to encode data : {:.03f}s'.format(time.time() - start_time))
    return X_enc, y_enc, mask


def Split_Data(X_set, y_set, val_size):
    X_enc, y_enc, mask = OH_Encoding(X_set, y_set)
    X_train, X_val, y_train, y_val = cross_validation.train_test_split(X_enc, y_enc, test_size=val_size, random_state=1260)
    mask_train, mask_val = cross_validation.train_test_split(mask, test_size=val_size, random_state=1260)
    return X_train, X_val, y_train, y_val, mask_train, mask_val


def batch_gen(X, y, Mask, N):
    while True:
        idx = np.random.choice(len(y), N)
        yield X[idx].astype('float32'), y[idx].astype('float32'), Mask[idx].astype('float32')


def LSTM(MAX_LENGTH, N_HIDDEN1):
    l_in = lasagne.layers.InputLayer(shape=(None, None, 1))

    l_mask = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH))

    gate_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(), b=lasagne.init.Constant(0.))

    cell_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(), W_cell=None, b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.tanh)

    l_lstm1 = lasagne.layers.recurrent.LSTMLayer(l_in, N_HIDDEN1, mask_input=l_mask, ingate=gate_parameters, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, learn_init=True, grad_clipping=100.)

    l_lstm1_back = lasagne.layers.recurrent.LSTMLayer(l_in, N_HIDDEN1, ingate=gate_parameters, mask_input=l_mask, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, learn_init=True, grad_clipping=100., backwards=True)

    l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm1, l_lstm1_back])

    l_lstm2 = lasagne.layers.recurrent.LSTMLayer(l_sum, N_HIDDEN1, mask_input=l_mask, ingate=gate_parameters, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, learn_init=True, grad_clipping=100.)

    l_lstm2_back = lasagne.layers.recurrent.LSTMLayer(l_sum, N_HIDDEN1, ingate=gate_parameters, mask_input=l_mask, forgetgate=gate_parameters, cell=cell_parameters, outgate=gate_parameters, learn_init=True, grad_clipping=100., backwards=True)

    l_sum = lasagne.layers.ElemwiseSumLayer([l_lstm2, l_lstm2_back])

    l_lstm_slice = lasagne.layers.SliceLayer(l_sum, 0, 1)

    l_out = lasagne.layers.DenseLayer(l_lstm_slice, num_units=2, nonlinearity=lasagne.nonlinearities.sigmoid)

    return l_in, l_mask, l_out


def Train_model(BATCH_SIZE, number_of_epochs, lr):
    X_train, X_val, y_train, y_val, mask_train, mask_val = Split_Data(Words, labels, val_size=0.3)

    l_in, l_mask, l_out = LSTM(MAX_LENGTH=mask_train.shape[1], N_HIDDEN1=128)

    y_sym = T.matrix()
    mask = T.matrix()

    output = lasagne.layers.get_output(l_out)
    pred = (output > 0.5)

    loss = T.mean(lasagne.objectives.binary_crossentropy(output, y_sym))

    acc = T.mean(T.eq(pred, y_sym))

    params = lasagne.layers.get_all_params(l_out)
    grad = T.grad(loss, params)
    updates = lasagne.updates.adam(grad, params, learning_rate=lr)

    f_train = theano.function([l_in.input_var, y_sym, l_mask.input_var], [loss, acc], updates=updates)
    f_val = theano.function([l_in.input_var, y_sym, l_mask.input_var], [loss, acc])
    f_predict = theano.function([l_in.input_var, l_mask.input_var], pred)

    N_BATCHES = len(X_train) // BATCH_SIZE
    N_VAL_BATCHES = len(X_val) // BATCH_SIZE
    train_batches = batch_gen(X_train, y_train, mask_train, BATCH_SIZE)
    val_batches = batch_gen(X_val, y_val, mask_val, BATCH_SIZE)

    # Store learning state after each epoch into a list
    global predictions
    predictions = []

    print("Start training")
    for epoch in range(number_of_epochs):
        train_loss = 0
        train_acc = 0
        start_time = time.time()
        for _ in range(N_BATCHES):
            X, y, mask = next(train_batches)
            loss, acc = f_train(X, y, mask)
            predictions.append(f_predict(X, mask))
            train_loss += loss
            train_acc += acc
        train_loss /= N_BATCHES
        train_acc /= N_BATCHES

        val_loss = 0
        val_acc = 0
        for _ in range(N_VAL_BATCHES):
            X, y, mask = next(val_batches)
            loss, acc = f_val(X, y, mask)
            val_loss += loss
            val_acc += acc
        val_loss /= N_VAL_BATCHES
        val_acc /= N_VAL_BATCHES
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, number_of_epochs, time.time() - start_time))
        print('  Train loss: {:.03f} - Validation Loss: {:.03f}'.format(
            train_loss, val_loss))
        print('  Train accuracy: {:.03f}'.format(train_acc))
        print('  Validation accuracy: {:.03f}'.format(val_acc))
        # np.savez('model.npz', *lasagne.layers.get_all_param_values(l_out))


if __name__ == "__main__":
    Train_model(BATCH_SIZE=32, number_of_epochs=10, lr=0.0005)
    del Words, labels 