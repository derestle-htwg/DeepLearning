import numpy as np

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
        return 1.0/(1.0 + np.exp(-float(t)))

class Layer:
    neurons = []
    def __init__(self, neuronCount, prevLayer):
        self.prevLayer = prevLayer
        if prevLayer != 0:
            prevLayer.nextLayer = self
        for i in range(neuronCount):
            self.neurons = Neuron(prevLayer)
        A = []

    def forward(self):
        for i in range(len(self.neurons)):
            self.neurons[i].calc()
            self.A[i] = self.neurons[i].A


class Neuron:
    def __init__(self, prevLayer):
        self.prevLayer = prevLayer
        if prevLayer == 0:
            A = 0
        else:
            W = np.ones(len(prevLayer.neurons))
            B = 0.0

    def calc(self):
        if self.prevLayer == 0:
            return

        # Gewichtete Summe d. vorherigen neuronen berechnen
        tmp = 0.0
        for i in len(self.W):
            tmp += tmp + self.w[i]* self.prevLayer.neurons[i].A
        self.L = tmp + self.B
        self.A = sig(self.L)

    def setA(self, value):
        self.A = value




    E1 = [0.0, 0.0]
    E2 = [1.0, 0.0]
    E3 = [0.0, 1.0]
    E4 = [1.0, 1.0]

    Y = [[0.0, 0.0, 0.0],
         [0.0, 1.0, 1.0],
         [0.0, 1.0, 1.0],
         [1.0, 1.0, 0.0]]

    E = [E1,E2,E3,E4]

    def NN(self, E):

        print('Input: ')
        print(E)
        w = [[10.0,10.0],[10.0,10.0]]
        b = [-15.0, -5.0]
        b_2 = [-4.0, -4.0, -4.0]
        H = np.dot(E, w) + b
        print(H)

        for x in np.nditer(H, op_flags=['readwrite']):
            x[...] = sig(x)

        print(H)

        v = [[10.0,0.0,-20.0],[0.0,10.0,10.0]]
        A =np.dot(H, v) + b_2
        print(A)

        for x in np.nditer(A, op_flags=['readwrite']):
            x[...] = sig(x)

        print(A)

        for x in np.nditer(A, op_flags=['readwrite']):
            if x >= 0.5:
                x[...] = 1
            else:
                x[...] = 0

        print('Output: And/Or/Xor ')
        print(A)


for e in E:
    print("******************")
    nn(e)

