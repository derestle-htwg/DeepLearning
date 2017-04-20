# -*- coding: utf-8 -*-
import numpy as np

alpha = 1

def gradA(lNext, index):

    w_j1 = np.zeros([1, len(lNext.neurons)])
    gradA_j1 = np.zeros([1, len(lNext.neurons)])
    l_j1 = np.zeros([1, len(lNext.neurons)])
    for i in range(len(lNext.neurons)):
        tmp = lNext.neurons[i].W[0, index]
        w_j1[0, i] = tmp
        tmp = lNext.neurons[i].dA
        gradA_j1[0, i] = tmp
        tmp = lNext.neurons[i].L
        l_j1[0, i] = tmp

    tmp = np.dot(np.dot(np.matrix(gradSig(l_j1)), np.matrix(gradA_j1)), np.matrix(w_j1.T))
    return tmp
#np.dot(np.dot(np.matrix(w_j1), np.matrix(gradA_j1)), np.matrix(gradSig(l_j1.T)))

def gradB(deltA_j, L_j):
    return np.dot(deltA_j, gradSig(L_j))

def gradW(deltA_j, L_j, A_jm1):
    return np.dot(np.dot(deltA_j, gradSig(L_j)).T, A_jm1.T)
    #return np.dot(gradB(deltA_j, L_j), A_jm1)

def sig(t):
        return 1.0/(1.0 + np.exp(t))

def gradSig(t):
        return np.multiply(sig(t), (1-sig(t)))

class Layer:
    def __init__(self, neuronCount, prevLayer):
        self.neurons = []*neuronCount
        self.A = np.zeros((1,neuronCount))
        self.L = np.zeros((1,neuronCount))
        self.prevLayer = prevLayer
        self.nextLayer = 0
        if self.prevLayer != 0:
            self.prevLayer.nextLayer = self
        for i in range(neuronCount):
            self.neurons.append(Neuron(prevLayer))


    def forward(self):
        for i in range(len(self.neurons)):
            self.neurons[i].calc()
            self.A[0, i] = self.neurons[i].A
            self.L[0, i] = self.neurons[i].L


class Neuron:
    def __init__(self, prevLayer):
        self.prevLayer = prevLayer
        self.dB = 0.0
        self.L = 0.0
        self.A = 0.0
        self.B = 0.0
        self.dW = np.zeros((1, 1))
        self.W = np.zeros((1, 1))
        self.dA = 0
        if prevLayer != 0:
            self.dW = np.zeros(prevLayer.A.shape)
            self.W = np.random.rand(prevLayer.A.shape[0], prevLayer.A.shape[1])

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

In = Layer(2, 0)
Mid = Layer(2, In)
Out = Layer(3, Mid)

#Mid.neurons[0].W[0, 0] = (10)
#Mid.neurons[0].W[0, 1] = (10)
#Mid.neurons[0].B = -15
#Mid.neurons[1].W[0,0] = (10)
#Mid.neurons[1].W[0,1] = (10)
#Mid.neurons[1].B = -5
#
#Out.neurons[0].W[0,0] = (10)
#Out.neurons[0].W[0,1] = (0)
#Out.neurons[0].B = -4
#
#Out.neurons[1].W[0,0] = (0)
#Out.neurons[1].W[0,1] = (10)
#Out.neurons[1].B = -4
#
#Out.neurons[2].W[0,0] = (-20)
#Out.neurons[2].W[0,1] = (10)
#Out.neurons[2].B = -4

def backPropNN(E,Layers,Y):

    for i in range(len(Layers[-1].neurons)):
        Layers[-1].neurons[i].dA = Y[i] - Layers[-1].A[0,i]
        Layers[-1].neurons[i].dB =  gradB(np.matrix(Layers[-1].neurons[i].dA), np.matrix(Layers[-1].L[0, i]))
                                    #dA*gradSig(Layers[-1].L[0,i])
        Layers[-1].neurons[i].dW = gradW(np.matrix(Layers[-1].neurons[i].dA),
                                         np.matrix(gradSig(Layers[-1].L[0, i])),
                                         np.matrix(Layers[-2].A))
                                        #np.dot(np.matrix(Layers[-1].neurons[i].dB),np.matrix(Layers[-2].A))
    print("backProp")
    for l in reversed(Layers[1:-1]):
        print("NewLayer")
        for i in range(len(l.neurons)):
            print("neuron ", i)
            n = l.neurons[i]
            n.dA = gradA(l.nextLayer, i)
            print(n.dA)
            n.dB = gradB(n.dA, n.L)
            print(n.dB)
            n.dW = gradW(n.dA, n.L, np.matrix(l.A))
            print(n.dW)


    for l in Layers[1:]:
        for n in l.neurons:

            n.B = n.B - (n.dB * alpha)
            n.W = n.W - (n.dW * alpha)

    return

def error(A,Y):
    tmp = 0
    for i in range(len(Y)):
        tmp += (Y[i]-A[i])**2
    return tmp

def forwardNN(E,Layers):
    i = 0
    # Eingabevektor in Eingabelayer schreiben
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
E = [E1, E2, E3, E4]
Y = [[0.0, 0.0, 0.0],
     [0.0, 1.0, 1.0],
     [0.0, 1.0, 1.0],
     [1.0, 1.0, 0.0]]
Layers = [In, Mid, Out]

#print(forwardNN(E1,Layers))
#print(forwardNN(E2,Layers))


for i in range(1000):
    x = np.random.random_integers(0, 3)
    A = forwardNN(E[x], Layers)
    #print(E[x])
    #print(A)
    #print(Y[x])
    #print()

    if (i % 1000) == 0:
        print(error(A[0], Y[x]))
    backPropNN(E[x], Layers, Y[x])

for x in range(4):
    A = forwardNN(E[x], Layers)
    print(error(A[0], Y[x]))
    print(A)
    print(Y[x])


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

