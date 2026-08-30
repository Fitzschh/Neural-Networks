import math
from operator import le

def dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    total = 0
    for i in range(len(v1)):
        total += v1[i] * v2[i]
    return total

def vector_mul(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same dimensions")
    res = [v1[i] * v2[i] for i in range(len(v1))]
    return res



def logits(A, x, b):
    if len(A[0]) != len(x):
        raise ValueError("Number of columns in A must match the number of rows in x")
    Ax = []
    for i in range(len(A)):
        row_sum = 0
        for j in range(len(A[i])):
            row_sum += A[i][j] * x[j]
        Ax.append(row_sum + b[i])
    return Ax

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
    exp_x = [2.71828182846 ** i for i in x]
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
    M_new = []
    for j in range(len(M[0])):
        row = []
        for i in range(len(M)):
            row.append(M[i][j])
        M_new.append(row)
    dLdx = []
    if len(M_new[0]) != len(x):
        raise ValueError("Must be of the same dimension")
    for i in range(len(M_new)):
        row_sum = 0
        for j in range(len(M_new[i])):
            row_sum += M_new[i][j] * x[j]
        dLdx.append(row_sum)
    return dLdx

def dReLU(x):
    dLdx = []
    for i in x:
        if i > 0:
            dLdx.append(1)
        else:
            dLdx.append(0)
    return dLdx

def gradient_descent(M, x, n):
    for i in range(len(M)):
        for j in range(len(x)):
            M[i][j] = M[i][j] - (n * x[j])
    return M




        


