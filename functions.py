import math
from PIL import Image
import numpy as np

def dot_product(v1, v2):
    return v1 @ v2

def vector_mul(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same dimensions")
    res = [v1[i] * v2[i] for i in range(len(v1))]
    return res

def logits(A, x, b):
    Ax = A @ x
    Ax_b = Ax + b
    return Ax_b

def activation(z):
    if z > 0:
        return z
    else:
        return 0

def summation(x):
    result = 0
    for i in x:
        result += i
    return result

def softmax(x):
    exp_x = [math.exp(i) for i in x]
    sum_exp_x = summation(exp_x)
    return [x / sum_exp_x for x in exp_x]

def cross_entropy_loss(pred, target):
    return -math.log(pred[target])

def hidden_layers(W, x, b):
    z = logits(W, x, b)
    a = [activation(z[i]) for i in range(len(z))]
    return a

def loss_gradients(p, y):
    if len(p) != len(y):
        raise ValueError("Both lists must be of the same dimensions")
    loss = []
    for i in range(len(p)):
        loss.append(p[i] - y[i])
    return loss

def backpropagation(M, x):
    #Gradient of the loss with respect to the weights
    return M.T @ x

def dReLU(x):
    dLdx = []
    for i in x:
        if i > 0:
            dLdx.append(1)
        else:
            dLdx.append(0)
    return dLdx

def gradient(dLdx, x):
    W = []
    for i in range(len(dLdx)):
        row = []
        for j in range(len(x)):
            row.append(dLdx[i] * x[j])
        W.append(row)
    return W
            
def gradient_descent(M, x, n):
    for i in range(len(M)):
        for j in range(len(x)):
            M[i][j] = M[i][j] - (n * x[i][j])
    return M

def gradient_descent_bias(b, x, n):
    if len(b) != len(x):
        raise ValueError("Vectors must be of the same dimensions")
    for i in range(len(b)):
        b[i] = b[i] - (n * x[i])
    return b

#Inputs

def image_to_vector(path):
    image = Image.open(path)

    image = image.convert("L")
    image = image.resize((28, 28))

    pixels = list(image.getdata())

    return [pixel / 255.0 for pixel in pixels]

def label_to_vector(label):
    vector = [0] * 10
    vector[label] = 1
    return vector








        


