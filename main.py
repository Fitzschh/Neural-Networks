from functions import image_to_vector
from training import train
from inputs import M, b, M2, b2, OL, b3, t, n, t2

file = open("images/t10k-images-idx3-ubyte", "rb")
data = file.read(16)
print(list(data))

image = file.read(784)
print(list(image))

pixels = list(image)


inputs = []

for pixel in pixels:
    inputs.append(pixel / 255)

print(inputs)

train(M, b, inputs, M2, b2, OL, b3, t, t2, n, 4)