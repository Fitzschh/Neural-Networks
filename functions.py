import math

def dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    total = 0
    for i in range(len(v1)):
        total += v1[i] * v2[i]
    return total

def matrix_vector_mul(A, x):
    if len(A[0]) != len(x):
        raise ValueError("Number of columns in A must match the number of rows in x")
    Ax = []
    for i in range(len(A)):
        row_sum = 0
        for j in range(len(A[i])):
            row_sum += A[i][j] * x[j]
        Ax.append(row_sum)
    return Ax

def activation(x, W, b):
    pre_act = dot_product(W, x) + b
    if pre_act > 0:
        return pre_act
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



