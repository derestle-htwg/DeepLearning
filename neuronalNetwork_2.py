from unittest.test.testmock.testpatch import something

import numpy as np
from sklearn import neural_network


def gradSig(x):
    #for x in np.nditer(L, op_flags=['readwrite']):
    return sig(x)*(1 - sig(x))

def gradA(W_j1, gradA_j1, L_j1 ):
    return np.dot(np.dot(W_j1, gradA_j1), gradSig(L_j1))

def gradB(deltA_j, L_j):
    return np.dot(deltA_j, gradSig(L_j))

def gradW(deltA_j, L_j, A_jm1):
    return np.dot(gradB(deltA_j, L_j), A_jm1)

def sig(t):
        return 1.0/(1.0 + np.exp(t))

def gradSig(t):
        return sig(t)*(1-sig(t))

class Layer:
    def __init__(self, neuronCount, prevLayer):
        self.neurons = []*neuronCount
        self.A = np.zeros((1,neuronCount))
        self.L = np.zeros((1,neuronCount))
        self.prevLayer = prevLayer
        if self.prevLayer != 0:
            self.prevLayer.nextLayer = self
        for i in range(neuronCount):
            self.neurons.append(Neuron(prevLayer))


    def forward(self):
        for i in range(len(self.neurons)):
            self.neurons[i].calc()
            self.A[0,i] = self.neurons[i].A
            self.L[0,i] = self.neurons[i].L


class Neuron:
    def __init__(self, prevLayer):
        self.prevLayer = prevLayer
        self.dB = 0.0
        self.L = 0.0
        self.A = 0.0
        self.B = 0.0
        self.dW = np.zeros((1,1))
        self.W = np.zeros((1,1))
        if prevLayer != 0:
            self.dW = np.zeros(prevLayer.A.shape)
            self.W = np.zeros(prevLayer.A.shape)

    def calc(self):
        if self.prevLayer == 0:
            return

        # Gewichtete Summe d. vorherigen neuronen berechnen
        tmp = 0.0
        for i in range(np.shape(self.W)[0]):
            tmp += self.W[0,i] * self.prevLayer.A[0,i]
        self.L = tmp + self.B
        self.A = sig(self.L)

    def setA(self, value):
        self.A = value

In = Layer(2,0)
Mid = Layer(2,In)
Out = Layer(3,Mid)

Mid.neurons[0].W[0,0] = (10)
Mid.neurons[0].W[0,1] = (10)
Mid.neurons[0].B = -15
Mid.neurons[1].W[0,0] = (10)
Mid.neurons[1].W[0,1] = (10)
Mid.neurons[1].B = -5

Out.neurons[0].W[0,0] = (10)
Out.neurons[0].W[0,1] = (0)
Out.neurons[0].B = -4

Out.neurons[1].W[0,0] = (0)
Out.neurons[1].W[0,1] = (10)
Out.neurons[1].B = -4

Out.neurons[2].W[0,0] = (-20)
Out.neurons[2].W[0,1] = (10)
Out.neurons[2].B = -4

def backPropNN(E,Layers,Y):

    for i in range(len(Layers[-1].neurons)):
        dA = Layers[-1].A[0,i] - Y[i]
        Layers[-1].neurons[i].dB = dA*gradSig(Layers[-1].L[0,i])
        Layers[-1].neurons[i].dW = np.dot(np.matrix(Layers[-1].neurons[i].dB),np.matrix(Layers[-2].A))
        Layers[-1].neurons[i].W = np.add(Layers[-1].neurons[i].W,Layers[-1].neurons[i].dW)
        Layers[-1].neurons[i].B = Layers[-1].neurons[i].B + Layers[-1].neurons[i].dB

    for l in reversed(Layers):
        for n in l.neurons:
            dA = Layers[-1].A[0,i] - Y[i]
            l.neurons[i].dB = dA*gradSig(l.L[0,i])
            l.neurons[i].dW = np.dot(np.matrix(l.neurons[i].dB),np.matrix(Layers[-2].A))
            l.neurons[i].W = np.add(Layers[-1].neurons[i].W,Layers[-1].neurons[i].dW)
            l.neurons[i].B = Layers[-1].neurons[i].B + Layers[-1].neurons[i].dB



def forwardNN(E,Layers):
    i = 0
    #Eingabevektor in Eingabelayer schreiben
    for e in E:
        Layers[0].neurons[i].A = e
        i = i + 1
    # Layer für Layer aus dem vorherigen layer berechnen
    for l in Layers:
        l.forward()

    return Layers[-1].A

E1 = [0.0, 0.0]
E2 = [1.0, 0.0]
E3 = [0.0, 1.0]
E4 = [1.0, 1.0]
Y = [[0.0, 0.0, 0.0],
     [0.0, 1.0, 1.0],
     [0.0, 1.0, 1.0],
     [1.0, 1.0, 0.0]]
Layers = [In, Mid, Out]

#print(forwardNN(E1,Layers))
#print(forwardNN(E2,Layers))
print(forwardNN(E3,Layers))
backPropNN(E3,Layers,Y[2])
print(forwardNN(E3,Layers))
backPropNN(E3,Layers,Y[2])
print(forwardNN(E3,Layers))
backPropNN(E3,Layers,Y[2])
#print(forwardNN(E4,Layers))



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

