from functions import hidden_layers, softmax, logits
from inputs import M, b, seven, M2, b2, OL, b3

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