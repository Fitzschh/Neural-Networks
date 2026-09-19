from functions import image_to_vector, label_to_vector
from training import train, predict, batch_gradient, batch_descent
from inputs import M, b, M2, b2, OL, b3, n, t2
import matplotlib.pyplot as plt
import time
import random

random.seed(42)

file = open("images/train-images-idx3-ubyte", "rb")
data = file.read(16)

file2 = open("images/train-labels-idx1-ubyte", "rb")
data2 = file2.read(8)

file3 = open("images/t10k-images-idx3-ubyte", "rb")
data3 = file3.read(16)

file4 = open("images/t10k-labels-idx1-ubyte", "rb")
data4 = file4.read(8)

images = []
labels = []
epochs = []
avg_loss = []
accuracy_list = []
training_times = []

training_images = 20000
batch_size = 32

for j in range(training_images):
    image = file.read(784)
    label = file2.read(1)

    images.append(image)
    labels.append(label[0])


n = 0.1

training_start = time.perf_counter()

for epoch in range(20):

    total_loss = 0

    #Shuffling images and labels
    indices = list(range(len(images)))

    random.shuffle(indices)

    for start in range(0, len(images), batch_size):

        batch_indices = indices[start:start + batch_size]

        W_gradient_list = []
        b_gradient_list = []
        W2_gradient_list = []
        b2_gradient_list = []
        OL_gradient_list = []
        b3_gradient_list = []

        for i in batch_indices:

            target = label_to_vector(labels[i])
            #print(target)
            pixels = list(images[i])
            inputs = []
                    
            for pixel in pixels:
                inputs.append(pixel / 255)

            dLdW, dLdb, dLdW2, dLdb2, dLdOL, dLdb3, loss = batch_gradient(M, b, M2, b2, OL, b3, inputs, labels[i], target, 1)
            total_loss += loss
            W_gradient_list.append(dLdW)
            b_gradient_list.append(dLdb)
            W2_gradient_list.append(dLdW2)
            b2_gradient_list.append(dLdb2)
            OL_gradient_list.append(dLdOL)
            b3_gradient_list.append(dLdb3)

        #Averaging the gradients of W
        avg_gradient_W = []

        for row in range(128):

            avg_row = []

            for col in range(784):

                gradient_sum = 0

                for g in W_gradient_list:
                    gradient_sum += g[row][col]

                average = gradient_sum / len(W_gradient_list)

                avg_row.append(average)

            avg_gradient_W.append(avg_row)

        #Averaging the gradients of b
        avg_gradient_b = [sum(g[i] for g in b_gradient_list) / len(b_gradient_list) for i in range(128)]

        avg_gradient_W2 = []

        for row in range(64):

            avg_row = []

            for col in range(128):

                gradient_sum = 0

                for g in W2_gradient_list:
                    gradient_sum += g[row][col]

                average = gradient_sum / len(W2_gradient_list)

                avg_row.append(average)

            avg_gradient_W2.append(avg_row)

        avg_gradient_b2 = [sum(g[i] for g in b2_gradient_list) / len(b2_gradient_list) for i in range(64)]

        avg_gradient_OL = []

        for row in range(10):
            
            avg_row = []

            for col in range(64):

                gradient_sum = 0

                for g in OL_gradient_list:
                    gradient_sum += g[row][col]

                average = gradient_sum / len(OL_gradient_list)

                avg_row.append(average)

            avg_gradient_OL.append(avg_row)

        avg_gradient_b3 = [sum(g[i] for g in b3_gradient_list) / len(b3_gradient_list) for i in range(10)]

        #Updating weights and biases
        M, b, M2, b2, OL, b3 = batch_descent(M, b, M2, b2, OL, b3, avg_gradient_W, avg_gradient_b, avg_gradient_W2, avg_gradient_b2, avg_gradient_OL, avg_gradient_b3, n)

    #n = n * 0.95

    average_loss = total_loss / len(images)

    print(f"Epoch {epoch + 1} --- Average Loss: {average_loss}")
    avg_loss.append(average_loss)
    epochs.append(epoch + 1)

training_time = time.perf_counter() - training_start
training_times.append(training_time)

print(f"Training time for {training_images} images: {training_time:.2f} seconds")

correct_pred = []
#Prediction Only
num_of_images = 10000

file3.seek(16)
file4.seek(8)

for i in range(num_of_images):
    image = file3.read(784)
    label = file4.read(1)

    label = label[0]
    target = label_to_vector(label)
    #print(f"Actual label: {label}")
    pixels = list(image)

    inputs = []

    for pixel in pixels:
        inputs.append(pixel / 255)

    p = predict(M, b, inputs, M2, b2, OL, b3)
    if p == label:
        correct_pred.append(1)

accuracy = (sum(correct_pred) / num_of_images) * 100
accuracy_list.append(accuracy)
print(f"Accuracy: {accuracy}%")



