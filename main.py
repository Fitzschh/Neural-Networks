from functions import image_to_vector, label_to_vector
from training import train, predict
from inputs import M, b, M2, b2, OL, b3, n, t2

file = open("images/t10k-images-idx3-ubyte", "rb")
data = file.read(16)

file2 = open("images/t10k-labels-idx1-ubyte", "rb")
data = file2.read(8)


images = []
labels = []
#Read images and labels
for i in range(9990):
    image = file.read(784)
    label = file2.read(1)

    images.append(image)
    labels.append(label[0])

#Training
for epoch in range(10):

    for i in range(len(images)):

        target = label_to_vector(labels[i])
        print(target)
        pixels = list(images[i])
        inputs = []
                
        for pixel in pixels:
            inputs.append(pixel / 255)

        M, b, M2, b2, OL, b3 = train(M, b, inputs, M2, b2, OL, b3, target, labels[i], n, 1)

#Prediction Only
for i in range(10):
    image = file.read(784)
    label = file2.read(1)

    label = label[0]
    target = label_to_vector(label)
    print(f"Actual label: {label}")
    pixels = list(image)

    inputs = []

    for pixel in pixels:
        inputs.append(pixel / 255)

    predict(M, b, inputs, M2, b2, OL, b3)

