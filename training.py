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
        dLdzi = loss_gradients(y, t)


        
