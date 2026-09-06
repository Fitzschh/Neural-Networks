from functions import image_to_vector, label_to_vector
from training import train
from inputs import M, b, M2, b2, OL, b3, n, t2

file = open("images/t10k-images-idx3-ubyte", "rb")
data = file.read(16)
print(list(data))

file2 = open("images/t10k-labels-idx1-ubyte", "rb")
data = file2.read(8)
print(list(data))

image = file.read(784)
print(list(image))
label = file2.read(1)
print(list(label))

label = label[0]

target = label_to_vector(label)

print(target)

pixels = list(image)

inputs = []

for pixel in pixels:
    inputs.append(pixel / 255)

train(M, b, inputs, M2, b2, OL, b3, target, label, n, 100)