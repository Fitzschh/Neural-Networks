from functions import hidden_layers, softmax, logits, cross_entropy_loss, loss_gradients, backpropagation, dReLU, vector_mul, gradient_descent, gradient_descent_bias, gradient
from inputs import M, b, seven, M2, b2, OL, b3, t, n

a = hidden_layers(M, seven, b)
print("First Layer")
print(a)

a2 = hidden_layers(M2, a, b2)
print("Second Layer")
print(a2)

z = logits(OL, a2, b3)
print("Output Layer")
print(z)

y = softmax(z)
print("Prediction")
print(y)

loss_7 = cross_entropy_loss(y, 7)
print(loss_7)

#Skipped Jacobian Matrix
loss_grad = loss_gradients(y, t)
print(loss_grad)

#Theory check
summed_grad = 0
for i in range(len(loss_grad)):
    summed_grad += loss_grad[i]
#Fundamentally should equate to zero

print("dL/da2")
dLda2 = backpropagation(OL, loss_grad)

da2dz2 = dReLU(a2)
print(da2dz2)

dLdz2 = vector_mul(dLda2, da2dz2)
print("Output Layer -> Second Layer")
print(dLdz2)
#Backpropagation from Output Layer to Second Layer

dLda = backpropagation(M2, dLdz2)
print("Second Layer")
print(dLda)

dadz = dReLU(a)
print(dadz)

dLdz = vector_mul(dLda, dadz)
print("Second Layer -> First Layer")
print(dLdz)
#Backpropagation from Second Layer to First Layer

dLdW = gradient(dLdz, seven)
print(dLdW)
M = gradient_descent(M, dLdW, n)
#Weights of First Layer altered"

dLdb = dLdz
b = gradient_descent_bias(b, dLdb, n)
print("Update bias")
print(b)
#Bias of First Layer updated"

dLdW2 = gradient(dLdz2, a)

M2 = gradient_descent(M2, dLdW2, n)
print("Second Layer weight updated")
print(M2)

dLdb2 = dLdz2

b2 = gradient_descent_bias(b2, dLdb2, n)
print("Second layer bias updated")
print(b2)
#Second Layer Backpropagation

dLdOL = gradient(loss_grad, a2)
OL = gradient_descent(OL, dLdOL, n)
print(OL)

dLdb3 = loss_grad
b3 = gradient_descent_bias(b3, dLdb3, n)
print("Output Layer bias updated")
print(b3)

