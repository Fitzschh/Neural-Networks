from functions import hidden_layers, softmax, logits, cross_entropy_loss, loss_gradients, backpropagation, dReLU, vector_mul, gradient_descent, gradient_descent_bias, gradient
from inputs import M, b, seven, M2, b2, OL, b3, t, n, t2


def train(M, b, input, M2, b2, OL, b3, t, t2, n, iter):

    for epoch in range(iter):
        #Forward propagation
        a = hidden_layers(M, input, b) #First Layer

        a2 = hidden_layers(M2, a, b2) #Second Layer

        z = logits(OL, a2, b3) #Output Layer

        y = softmax(z) 

        loss = cross_entropy_loss(y, t2)

        #Backpropagation
        dLdzi = loss_gradients(y, t)#Loss with respect to logits

        dLda2 = backpropagation(OL, dLdzi)#Loss with respect to activated values from Layer 2

        da2dz2 = dReLU(a2)#Derivative of ReLU in Layer 2

        dLdz2 = vector_mul(dLda2, da2dz2)#Backpropagation of Layer 2

        dLda = backpropagation(M2, dLdz2)#Loss with respect to activated values from Layer 1

        dadz = dReLU(a)#Derivative of ReLU in Layer 1

        dLdz = vector_mul(dLda, dadz)#Backpropagation of Layer 1

        dLdW = gradient(dLdz, input)#Distribute values of backpropagated outputs to each weights in Layer 1
        M = gradient_descent(M, dLdW, n)#Weights of 1st Layer altered

        dLdb = dLdz
        b = gradient_descent_bias(b, dLdb, n)#Biases of 1st Layer altered

        dLdW2 = gradient(dLdz2, a)#Distribute values of backpropagated outputs to each weights in Layer 2
        M2 = gradient_descent(M2, dLdW2, n)#Weights of 2nd Layer altered

        dLdb2 = dLdz2
        b2 = gradient_descent_bias(b2, dLdb2, n)#Biases of 2nd Layer altered

        dLdOL = gradient(dLdzi, a2)#Distribute values of backpropagated outputs to each weights in Output Layer
        OL = gradient_descent(OL, dLdOL, n)#Weights of Output Layer altered

        dLdb3 = dLdzi
        b3 = gradient_descent_bias(b3, dLdb3, n)#Biases of Output Layer altered

    int_pred = max(y)
    for i in range(len(y)):
        if y[i] == int_pred:
            pred = i

    return loss, iter, pred

loss, iterations, pred = train(M, b, seven, M2, b2, OL, b3, t, t2, n, 3)
print(loss, iterations, pred)













        
