from functions import hidden_layers, softmax, logits, cross_entropy_loss, loss_gradients, backpropagation, dReLU, vector_mul
from inputs import M, b, seven, M2, b2, OL, b3, t

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
print(dLda2)
print(len(dLda2))

da2dz2 = dReLU(a2)
print(da2dz2)

dLdz2 = vector_mul(dLda2, da2dz2)
print("Output Layer -> Second Layer")
print(dLdz2)
#Backpropagation from Output Layer to Second Layer

dLda = backpropagation(M2, dLdz2)
print("Second Layer")
print(dLda)

