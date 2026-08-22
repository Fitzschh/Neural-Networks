
A_1 = [
    [1, 2, -1, 3],
    [2, -1, 4, 0],
    [-1, 3, 2, 1]
]

v = [2, 1, -1, 3]
b_1 = [1, -2, 3]

A_2 = [
    [1, -2, 3],
    [2, 1, -1]
]

b_2 = [-1, 4]

target = 10

def MatrixVectorMul(A, x):
    if len(A[0]) != len(x):
        raise ValueError("Number of columns in a neuron must match the number of rows from the input")
    Ax = []
    for i in range(len(A)):
        row_sum = 0
        for j in range(len(A[i])):
            row_sum += x[j] * A[i][j]
        Ax.append(row_sum)
    return Ax

def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length")
    sum = 0
    for i in range(len(a)):
        sum += a[i] * b[i]
    return sum

def activation(x, W, b):
    pre_act = dot_product(W, x) + b
    if pre_act > 0:
        return pre_act
    else:
        return 0

def layers(x, W, b):
    z = MatrixVectorMul(W, x)
    for i in range(len(z)):
        z[i] = z[i] + b[i]
        if z[i] > 0:
            z[i] = z[i]
        else:
            z[i] = 0
    return z

def Loss(y_pred, target):
    loss = []
    for i in y_pred:
        loss.append(i - target)
    return loss

def PredictionGradient(y_pred, target):
    grad_loss = []
    for i in y_pred:
        grad_loss.append(2 ** (i - target))

a_1 = layers(v, A_1, b_1)
a_2 = layers(a_1, A_2, b_2)
print(a_1, a_2)

