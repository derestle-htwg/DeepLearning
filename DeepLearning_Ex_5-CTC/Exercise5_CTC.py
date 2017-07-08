import matplotlib.pyplot as plt
import numpy

def printProb_t(probabilities, str, idx):
    #print(probabilities)
    plt.subplot(2, 1, idx)
    plt.plot(probabilities[0].T, 'b-', label='blank')
    plt.plot(probabilities[1].T, 'r--', label='a')
    plt.plot(probabilities[2].T, 'g-', label='b')
    plt.plot(probabilities[3].T, 'y-', label='c')
    plt.legend(loc='upper right')
    plt.title(str + ' label probability')
    plt.xlabel("timesteps")
    plt.ylabel("label probability")
    #plt.axis([0, 11, 0, 1])
    #plt.show()

def printMatrixColored(matrix, str):
    c = plt.pcolor(matrix[::-1])
    plt.colorbar(c,orientation='horizontal')
    plt.winter()
    plt.title(str + ' probability over time')
    plt.xlabel("timesteps")
    plt.ylabel("label probability")
    plt.show()

def blankPadding(l_input):
    #2 * N +1
    size = 2*numpy.size(l_input)+1
    l_output = []
    l_output.append('-')
    idx = 1
    for c in l_input:
        l_output.append(c)
        l_output.append('-')
        idx += 2
    if (idx) != size :
        print("Error - Wrong size while padding: size = " + str(size) + ", idx = " + str(idx))
    return l_output

def forwardZeros(l_pad,T,t,s):
    #print(2*int(T - (t+1)))
    if (s < numpy.size(l_pad) - 2*int(T - (t+1)) - 1) | (s < 0):
        return 0.0
    else:
        return 1.0

def backwardZeros(l_pad,T,t,s):
    if (s > 2*int(t + 1)) | (s >= numpy.size(l_pad)):
        return 0.0
    else:
        return 1.0

def getRnnProb(l_pad, mat, t, s):
    char = l_pad[s]
    if char == '-':
        s2 = 0
    elif char == 'a':
        s2 = 1
    elif char == 'b':
        s2 = 2
    elif char == 'c':
        s2 = 3
    return mat[s2][t-1]

def logAdd(x1, x2):
    return numpy.log(float(x1)) + numpy.log(1 + numpy.exp(numpy.log(float(x2)) - numpy.log(float(x1))))

def logMul(x1, x2):
    return numpy.log(float(x1)) + numpy.log(float(x2))

def forwardRekursion(y_k_t, l_pad, T):
    # Calculate the forward chain
    alpha_t = numpy.zeros((numpy.size(l_pad), T))
    #print(alpha_t)

    # Initalization
    # alpha[s][t]
    alpha_t[0][0] = getRnnProb(l_pad, mat = y_k_t, t=1, s=0) #y_k_t[0][0]
    alpha_t[1][0] = getRnnProb(l_pad, mat = y_k_t, t=1, s=1) #y_k_t[1][0]
    #print(alpha_t)

    # Start dynamic-recursiv part
    for t in range(1, T):
        #print(t)

        for s in range(0, numpy.size(l_pad)):
            #print(s)

            currentChar = l_pad[s]
            alphaSum = 0.0

            if (currentChar == '-') | (currentChar == l_pad[s-2]):
                #print('Is in case: Blank or same label')
                if forwardZeros(l_pad,T,t - 1,s - 1) == 1.0:
                    #alphaSum = logAdd(alphaSum, alpha_t[s - 1][t - 1])
                    alphaSum += alpha_t[s - 1][t - 1]

                if forwardZeros(l_pad,T,t - 1,s) == 1.0:
                    #alphaSum = logAdd(alphaSum, alpha_t[s][t - 1])
                    alphaSum += alpha_t[s][t - 1]
#
            else:
                #print('Is in case: No blank or same label')
                if forwardZeros(l_pad, T, t - 1, s - 2) == 1.0:
                    #alphaSum = logAdd(alphaSum, alpha_t[s-2][t - 1])
                    alphaSum += alpha_t[s - 2][t - 1]

                if forwardZeros(l_pad, T, t - 1, s-1) == 1.0:
                    #alphaSum = logAdd(alphaSum, alpha_t[s-1][t - 1])
                    alphaSum += alpha_t[s-1][t - 1]

                if forwardZeros(l_pad, T, t - 1, s) == 1.0:
                    #alphaSum = logAdd(alphaSum, alpha_t[s][t - 1])
                    alphaSum += alpha_t[s][t - 1]

            #alpha_t[s][t] = logMul(alphaSum, getRnnProb(l_pad, mat=y_k_t, t=t, s=s))
            alpha_t[s][t] = alphaSum * getRnnProb(l_pad, mat = y_k_t, t=t, s=s)

    #return
    #print(alpha_t)
    return alpha_t

def backwardRekursion(y_k_t, l_pad, T):
    # Calculate the forward chain
    beta_t = numpy.zeros((numpy.size(l_pad), T))
    #print(alpha_t)

    # Initalization
    # alpha[s][t]
    beta_t[numpy.size(l_pad)-1][T-1] = 1.0
    beta_t[numpy.size(l_pad)-2][T-1] = 1.0
    #print(beta_t)

    # Start dynamic-recursiv part
    for t in range(0, T-1):
        # backward through time
        t = (T-t)-2
        #print(t)

        for s in range(0, numpy.size(l_pad)):
            #print(s)

            currentChar = l_pad[s]
            if s < numpy.size(l_pad)-2:
                overNext = l_pad[s+2]
            else:
                overNext = '-'
            betaSum = 0.0

            if (currentChar == '-') | (currentChar == overNext):
                #print('Is in case: Blank or same label')
                if backwardZeros(l_pad,T,t - 1,s) == 1.0:
                    betaSum += beta_t[s][t + 1] #* getRnnProb(l_pad, mat = y_k_t, t=t+1, s=s)

                if backwardZeros(l_pad,T,t + 1,s + 1) == 1.0:
                    betaSum += beta_t[s + 1][t + 1] #* getRnnProb(l_pad, mat = y_k_t, t=t+1, s=s+1)

            else:
                #print('Is in case: No blank or same label')
                if backwardZeros(l_pad, T, t + 1, s + 2) == 1.0:
                    betaSum += beta_t[s + 2][t + 1] #* getRnnProb(l_pad, mat = y_k_t, t=t+1, s=s+2)

                if backwardZeros(l_pad, T, t + 1, s + 1) == 1.0:
                    betaSum += beta_t[s + 1][t + 1] #* getRnnProb(l_pad, mat = y_k_t, t=t+1, s=s+1)

                if backwardZeros(l_pad, T, t + 1, s) == 1.0:
                    betaSum += beta_t[s][t + 1] #* getRnnProb(l_pad, mat = y_k_t, t=t+1, s=s)

            beta_t[s][t] = betaSum * getRnnProb(l_pad, mat = y_k_t, t=t, s=s)

    #return
    #print(beta_t)
    return beta_t

def matrixNormalization(matrix):
    C_t = 0.0
    for t in range(0, 12):
        C_t = numpy.sqrt(sum(matrix.T[t])**2)
        for s in range(0, numpy.size(matrix.T[t])):
            matrix.T[t][s] = numpy.sqrt(matrix.T[t][s]**2) / C_t
    return matrix

def forwardBackwardProduct(forwardPass, backwardPass, l_pad, T):
    totalProb = numpy.zeros((numpy.size(l_pad), T))
    for t in range(0, T):
        for s in range(0, numpy.size(l_pad)):
            totalProb[s][t] = forwardPass[s][t] * backwardPass[s][t]
    #print(totalProb)
    return totalProb

def getLabelProb(totalProbs, T):
    labelProbs = numpy.zeros(T)
    for t in range(0,T):
        labelProbs[t] = sum(totalProbs.T[t])
    return labelProbs

def estimatePositions(label):
    pos_blank = []
    pos_a = []
    pos_b = []
    pos_c = []

    for s in range(0, numpy.size(label)):
        char = label[s]
        if char == '-':
            pos_blank.append(s)
        elif char == 'a':
            pos_a.append(s)
        elif char == 'b':
            pos_b.append(s)
        elif char == 'c':
            pos_c.append(s)

    return [pos_blank, pos_a, pos_b, pos_c]

def derivate_for_y_k_t(totalProbabilities, y_k_t, labelPositions, T, labelSize):

    derivationOf_y = numpy.zeros((numpy.shape(labelPositions)[0], T))

    for t in range(0,T):

        for s in range(0, numpy.shape(labelPositions)[0]):
            sumOfTotalProb = 0
            for k in labelPositions[s]:
                sumOfTotalProb += totalProbabilities[k][t]
            if y_k_t[s][t] == 0:
                derivationOf_y[s][t] = 0
            else:
                derivationOf_y[s][t] = sumOfTotalProb / y_k_t[s][t]

    return derivationOf_y

def getLossDerivation(labelProb, derivationOfy, T):

    lossDerivation_y_k_t = numpy.zeros((4, T))

    for t in range(0, T):
        for s in range(0, 4):
            lossDerivation_y_k_t[s][t] = -(1/labelProb[t]) * derivationOfy[s][t]

    return lossDerivation_y_k_t

def getMaxProbSting(lossDerivation, T):
    maxProbString = []
    #print(lossDerivation)
    for t in range(0, T):

        max_t = numpy.max(lossDerivation.T[t])
        if lossDerivation.T[t][0] == max_t:
            maxProbString.append('-')
        elif lossDerivation.T[t][1] == max_t:
            maxProbString.append('a')
        elif lossDerivation.T[t][2] == max_t:
            maxProbString.append('b')
        elif lossDerivation.T[t][3] == max_t:
            maxProbString.append('c')
    return maxProbString

def removeDoubbleCharacters(maxProbString):
    finalString = []
    if maxProbString[0] != '-':
        finalString.append(maxProbString[0])
    for s in range(1, numpy.size(maxProbString)):
        if (maxProbString[s] != '-') & (maxProbString[s] != maxProbString[s - 1]):
            finalString.append(maxProbString[s])
    return finalString

if __name__ == "__main__":
    ######################################
    ## PROGRAMM START CTC LOSS FUNCTION ##
    ######################################

    y_k_t = numpy.array([   [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25]]).T

    y_k_t2 = numpy.array([[0.25, 0.75, 0.00, 0.00],
                          [0.25, 0.75, 0.00, 0.00],
                          [0.00, 0.50, 0.50, 0.00],
                          [0.00, 0.50, 0.50, 0.00],
                          [0.00, 0.00, 1.00, 0.00],
                          [0.50, 0.00, 0.50, 0.00],
                          [0.50, 0.00, 0.25, 0.00],
                          [0.25, 0.00, 0.50, 0.25],
                          [0.00, 0.00, 0.50, 0.50],
                          [0.00, 0.00, 0.50, 0.50],
                          [0.25, 0.00, 0.25, 0.50],
                          [0.25, 0.00, 0.00, 0.75]]).T


    # Control output for network-output
    #y_k_t = y_k_t2
    TIME_WINDOW_SIZE = numpy.size(y_k_t[0])
    #print(y_k_t)

    # print a-priori probabilities (RNN)
    printProb_t(y_k_t, 'Framewise', 1)

    # create inputstring - labeling l'
    l = ['a','b','b','c']
    l_pad = blankPadding(l)
    print(l_pad)
    LABEL_SIZE = numpy.size(l_pad)

    # create forward variables
    forwardPass = forwardRekursion(y_k_t, l_pad, T = TIME_WINDOW_SIZE)
    # normalization
    forwardPass = matrixNormalization(forwardPass)
    #printMatrixColored(forwardPass, 'Forward')

    # create backward variables
    backwardPass = backwardRekursion(y_k_t, l_pad, T = TIME_WINDOW_SIZE)
    # normalization
    backwardPass = matrixNormalization(backwardPass)
    #printMatrixColored(backwardPass, 'Backward')

    # product of forward and backward variables for total probability of l at time t
    totalProbabilities = forwardBackwardProduct(forwardPass, backwardPass, l_pad, TIME_WINDOW_SIZE)
    totalProbabilities = matrixNormalization(totalProbabilities)

    # probablility for l given x
    labelProb = getLabelProb(totalProbabilities, TIME_WINDOW_SIZE)
    #print(labelProb)

    # estimate positions af k in l_pad
    labelPositions = estimatePositions(l_pad)
    #print(labelPositions)

    # differentiate p(l|x) to y_k_t
    derivationOfy = derivate_for_y_k_t(totalProbabilities, y_k_t, labelPositions, TIME_WINDOW_SIZE, LABEL_SIZE)
    #print(derivationOfy)

    # Calcuate the loss derivation L nach y_k_t
    lossDerivation = getLossDerivation(labelProb, derivationOfy, TIME_WINDOW_SIZE)
    lossDerivation = matrixNormalization(lossDerivation)
    ctcPlt = printProb_t(lossDerivation, 'CTC', 2)
    #print(lossDerivation)

    # get string with maximum probability - path pi
    maxProbString = getMaxProbSting(lossDerivation, TIME_WINDOW_SIZE)

    # remove blanks and doubble characters
    print(removeDoubbleCharacters(maxProbString))
    plt.show()
    printMatrixColored(totalProbabilities, 'Total')


