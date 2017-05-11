from unittest.test.testmock.testpatch import something

import numpy as np
from sklearn import neural_network
alpha = -0.01


def gradSig(x):
    tmp = sig(x)
    return 1 - (np.multiply(tmp, tmp))
    # for x in np.nditer(L, op_flags=['readwrite']):
    # return np.multiply(sig(x), (1 - sig(x)))


def gradA(w_j1, grada_j1, l_j1):
    return np.dot(np.dot(w_j1, grada_j1), gradSig(l_j1))


def gradB(deltA_j, l_j):
    return np.dot(deltA_j, gradSig(l_j))


def gradW(deltA_j, L_j, A_jm1):
    return np.dot(gradB(deltA_j, L_j), A_jm1)


def sig(t):
        #return 1.0/(1.0 + np.exp(t))
    return np.tanh(t)


class Layer:
    def __init__(self, neuronCount, prevLayer):
        self.dA = np.zeros((neuronCount, 1))
        self.dB = np.zeros((neuronCount, 1))
        self.dW = np.zeros((neuronCount, 1))
        self.A = np.zeros((neuronCount, 1))
        self.L = np.zeros((neuronCount, 1))
        self.B = np.random.rand(neuronCount, 1)
        self.W = np.zeros((0, 0))
        self.prevLayer = prevLayer
        self.nextLayer = 0
        if self.prevLayer != 0:
            self.W = np.random.rand(neuronCount, prevLayer.A.shape[0])
            self.prevLayer.nextLayer = self

    def forward(self):
        if self.prevLayer != 0:
            self.L = self.W.dot(self.prevLayer.A) + self.B
            self.A = sig(self.L)


In = Layer(2, 0)
Mid = Layer(2, In)
Out = Layer(3, Mid)
Layers = [In, Mid, Out]

def backPropNN(E, Layers, Y):

    dA = Layers[-1].A - Y
    Layers[-1].dB = np.multiply(dA, gradSig(Layers[-1].L))
    Layers[-1].dW = np.dot(Layers[-1].dB, Layers[-2].A.T)

    for l in reversed(Layers[1:-1]):
        l.dA = l.nextLayer.W.T.dot(l.nextLayer.dB)
        l.dB = np.multiply(l.dA, gradSig(l.L))
        l.dW = l.dB.dot(l.prevLayer.A.T)

    i = 0
    for l in Layers[1:]:
        i = i + 1
        # print(i)
        # print(l.dW)
        # print(l.dB)
        l.W = l.W + l.dW * alpha

        l.B = l.B + l.dB * alpha


def forwardNN(E,Layers):
    i = 0
    # Eingabevektor in Eingabelayer schreiben
    Layers[0].A = E

    # Layer für Layer aus dem vorherigen layer berechnen
    for l in Layers:
        l.forward()

    return Layers[-1].A


def error(A, Y):
    tmp = 0
    for i in range(len(Y)):
        tmp += (Y[i]-A[i])**2
    return tmp/len(A)

E1 = np.matrix([[0.0], [0.0]])
E2 = np.matrix([[1.0], [0.0]])
E3 = np.matrix([[0.0], [1.0]])
E4 = np.matrix([[1.0], [1.0]])
Y = [np.matrix([[0.0], [0.0], [0.0]]),
     np.matrix([[0.0], [1.0], [1.0]]),
     np.matrix([[0.0], [1.0], [1.0]]),
     np.matrix([[1.0], [1.0], [0.0]])]

E = [E1, E2, E3, E4]


for i in range(3000000):
    x = np.random.randint(0,4)
    forwardNN(E[x], Layers)
    backPropNN(Out.A, Layers, Y[x])
    if (i % 10000) == 0:

        print("Errors", i/1000)
        print(error(forwardNN(E[0], Layers), Y[0])[0, 0])
        print(error(forwardNN(E[1], Layers), Y[1])[0, 0])
        print(error(forwardNN(E[2], Layers), Y[2])[0, 0])
        print(error(forwardNN(E[3], Layers), Y[3])[0, 0])
        print(forwardNN(E[0], Layers))
        print(forwardNN(E[1], Layers))
        print(forwardNN(E[2], Layers))
        print(forwardNN(E[3], Layers))


#
#    E = [E1,E2,E3,E4]
#
#    def NN(self, E):
#
#        print('Input: ')
#        print(E)
#        w = [[10.0,10.0],[10.0,10.0]]
#        b = [-15.0, -5.0]
#        b_2 = [-4.0, -4.0, -4.0]
#        H = np.dot(E, w) + b
#        print(H)
#
#        for x in np.nditer(H, op_flags=['readwrite']):
#            x[...] = sig(x)
#
#        print(H)
#
#        v = [[10.0,0.0,-20.0],[0.0,10.0,10.0]]
#        A =np.dot(H, v) + b_2
#        print(A)
#
#        for x in np.nditer(A, op_flags=['readwrite']):
#            x[...] = sig(x)
#
#        print(A)
#
#        for x in np.nditer(A, op_flags=['readwrite']):
#            if x >= 0.5:
#                x[...] = 1
#            else:
#                x[...] = 0
#
#        print('Output: And/Or/Xor ')
#        print(A)
#
#
#for e in E:
#    print("******************")
#    nn(e)

