from transformation import layers, MatrixVectorMul, Loss, PredictionGradient

p = [
    [3, 2],
    [1, 3],
    [3, 1],
    [1, 1]
]

W = [
    [1, 1],
    [-1, 1],
    [1, -1]
]

b = [-2, -1, -1]

target = 10

a_1 = []

W_2 = [
    [1, 2, -1],
    [-2, 1, 1]
]

b_2 = [-1, 0]

a_2 = []

print("Activated Outputs")
for i in range(len(p)):
    x = p[i]
    a = layers(x, W, b)
    print(f"Input: {x}, Output: {a}")
    a_1.append(a)

print("Pre-Activation Outputs")
for i in range(len(p)):
    z = MatrixVectorMul(W, p[i])
    for j in range(len(z)):
        z[j] = z[j] + b[j]
    print(f"Input: {p[i]}, Output: {z}")

print("Layer 2")
for i in range(len(a_1)):
    a = layers(a_1[i], W_2, b_2)
    a_2.append(a)

for i in a_2:
    print(f"Activation Output: {i}")

L = Loss(a_2, target)

print(L)

