from functions import image_to_vector, label_to_vector
from training import train, predict, batch_gradient, batch_descent
from inputs import M, b, M2, b2, OL, b3, n
import matplotlib.pyplot as plt
import time
import random
import numpy as np

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

avg_loss = []
training_times = []

# Stores the epochs where accuracy was measured
epoch_points = []

# Stores accuracy measured at those epochs
accuracy_list = []

training_images = 60000
batch_size = 32

for j in range(training_images):
    image = file.read(784)
    label = file2.read(1)

    images.append(image)
    labels.append(label[0])

x_train = np.frombuffer(b''.join(images),dtype=np.uint8).astype(np.float32) / 255

x_train = x_train.reshape(training_images, 784)

n = 0.1

training_start = time.perf_counter()

for epoch in range(20):

    total_loss = 0

    # Shuffling images and labels
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

            inputs = x_train[i]

            dLdW, dLdb, dLdW2, dLdb2, dLdOL, dLdb3, loss = batch_gradient(M, b, M2, b2, OL, b3, inputs, labels[i], target, 1)

            total_loss += loss

            W_gradient_list.append(dLdW)
            b_gradient_list.append(dLdb)

            W2_gradient_list.append(dLdW2)
            b2_gradient_list.append(dLdb2)

            OL_gradient_list.append(dLdOL)
            b3_gradient_list.append(dLdb3)

        # Averaging the gradients of W
        avg_gradient_W = np.mean(W_gradient_list, axis=0)

        # Averaging the gradients of b
        avg_gradient_b = np.mean(b_gradient_list, axis=0)

        avg_gradient_W2 = np.mean(W2_gradient_list, axis=0)

        avg_gradient_b2 = np.mean(b2_gradient_list, axis=0)

        avg_gradient_OL = np.mean(OL_gradient_list, axis=0)

        avg_gradient_b3 = np.mean(b3_gradient_list, axis=0)

        # Updating weights and biases
        M, b, M2, b2, OL, b3 = batch_descent(M, b, M2, b2, OL, b3, avg_gradient_W, avg_gradient_b, avg_gradient_W2, avg_gradient_b2, avg_gradient_OL, avg_gradient_b3, n)

    # Learning-rate decay
    #n = n * 0.95

    average_loss = total_loss / len(images)

    print(f"Epoch {epoch + 1} --- Average Loss: {average_loss}")

    avg_loss.append(average_loss)

    if epoch + 1 >= 20:

        correct_pred = []

        num_of_images = 10000

        file3.seek(16)
        file4.seek(8)

        for i in range(num_of_images):

            image = file3.read(784)
            label = file4.read(1)

            label = label[0]

            inputs = np.frombuffer(image, dtype=np.uint8).astype(np.float32) / 255

            p = predict(M, b, inputs, M2, b2, OL, b3)

            if p == label:
                correct_pred.append(1)

        accuracy = (sum(correct_pred) / num_of_images) * 100

        epoch_points.append(epoch + 1)
        accuracy_list.append(accuracy)

        print(f"Accuracy after Epoch {epoch + 1}: {accuracy}%")

training_time = time.perf_counter() - training_start

training_times.append(training_time)

print(f"Training time for {training_images} images over 30 epochs: {training_time:.2f} seconds")

# Plot Epoch vs Accuracy
plt.plot(epoch_points, accuracy_list)
plt.title("Epochs vs Accuracy for 60k Images Training")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()