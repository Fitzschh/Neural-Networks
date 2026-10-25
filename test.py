from functions import image_to_vector, label_to_vector, activation, batch_softmax
from training import save_model, train, predict, batch_gradient, batch_descent
from inputs import M, b, M2, b2, OL, b3, n
from model_io import save_model, load_model
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

for epoch in range(24):

    total_loss = 0

    # Shuffling images and labels
    indices = list(range(len(images)))

    random.shuffle(indices)

    for start in range(0, len(images), batch_size):

        batch_indices = indices[start:start + batch_size]

        inputs = x_train[batch_indices]

        batch_labels = labels[batch_indices]

        target = np.zeros((batch_size, 10))
        target[np.arange(batch_size), batch_labels] = 1

        z = inputs @ M.T + b

        a = activation(z)

        z2 = a @ M2.T + b2

        a2 = activation(z2)

        z3 = a2 @ OL.T + b3

        y = batch_softmax(z3)
        


    # Learning-rate decay
    #n = n * 0.95

    average_loss = total_loss / len(images)

    print(f"Epoch {epoch + 1} --- Average Loss: {average_loss}")

    avg_loss.append(average_loss)

correct_pred = []
#Prediction only
num_of_images = 10000

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

accuracy = (sum(correct_pred) / num_of_images) * 100

training_time = time.perf_counter() - training_start

training_times.append(training_time)

print(f"Training time for {training_images} images over {epoch + 1} epochs: {training_time:.2f} seconds")
print(f"Accuracy: {accuracy:.2f}%")

while True:
    choice = input("Save trained model? (y/n): ").lower()
    if choice == "y":
        model_name = input("Enter model name: ")
        save_model(M, b, M2, b2, OL, b3, model_name)
        break
    elif choice == "n":
        break
    else:
        print("Invalid input")

        
