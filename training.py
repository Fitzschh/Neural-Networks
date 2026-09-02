from functions import hidden_layers, softmax, logits, cross_entropy_loss, loss_gradients, backpropagation, dReLU, vector_mul, gradient_descent, gradient_descent_bias, gradient
from inputs import M, b, seven, M2, b2, OL, b3, t, n


def train(M, b, seven, M2, b2, OL, b3, t, n, iter):

    for epoch in range(iter):
        #Forward propagation
        a = hidden_layers(M, seven, b) #First Layer

        a2 = hidden_layers(M2, a, b2) #Second Layer

        z = logits(OL, a2, b3) #Output Layer

        y = softmax(z) 

        #Backpropagation
        dLdzi = loss_gradients(y, t)#Loss with respect to logits

        dLda2 = backpropagation(OL, dLdzi)#Loss with respect to activated values from Layer 2

        da2dz2 = dReLU(a2)#Derivative of ReLU in Layer 2

        dLdz2 = vector_mul(dLda2, da2dz2)#Backpropagation of Layer 2

        dLda = backpropagation(M2, dLdz2)#Loss with respect to activated values from Layer 1

        dadz = dReLU(a)#Derivative of ReLU in Layer 1

        dLdz = vector_mul(dLda, dadz)#Backpropagation of Layer 1

        dLdW = gradient(dLdz, seven)#Distribute values of backpropagated outputs to each weights in Layer 1

        








        
