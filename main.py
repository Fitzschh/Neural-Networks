from functions import hidden_layers
from inputs import M, b, seven, M2, b2

a = hidden_layers(M, seven, b)
print("First Layer")
print(a)

a2 = hidden_layers(M2, a, b2)
print("Second Layer")
print(a2)



