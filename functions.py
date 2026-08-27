import math
from operator import le

def dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    total = 0
    for i in range(len(v1)):
        total += v1[i] * v2[i]
    return total

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
    max_x = max(x)
    exp_x = [2.71828182846 ** (i - max_x) for i in x]
    sum_exp_x = summation(exp_x)
    return [x / sum_exp_x for x in exp_x]

def cross_entropy_loss(pred, target):
    return -math.log(pred[target])

def hidden_layers(W, x, b):
    z = logits(W, x, b)
    a = [activation(z[i]) for i in range(len(z))]
    return a




