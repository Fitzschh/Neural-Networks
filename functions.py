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
    return max(0, z)

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

def cross_entropy_loss(pred, target):
    return -math.log(pred[target])

def hidden_layers(W, x, b):
    z = logits(W, x, b)
    return activation(z)

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
    return (x > 0).astype(float)

def gradient(dLdx, x):
    return np.outer(dLdx, x)
            
def gradient_descent(M, x, n):
    return M - (n * x)

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








        


