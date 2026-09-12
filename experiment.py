from functions import image_to_vector, label_to_vector
from training import train, predict
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

training_images = [10000, 20000, 30000, 40000, 50000, 60000]
previous_count = 0

for stage in range(len(training_images)):

    additional_images = training_images[stage] - previous_count

    for j in range(additional_images):
        image = file.read(784)
        label = file2.read(1)

        images.append(image)
        labels.append(label[0])

    previous_count = training_images[stage]

    training_start = time.perf_counter()

    n = 0.001

    for epoch in range(20):

        total_loss = 0

        #Shuffling images and labels
        indices = list(range(len(images)))

        random.shuffle(indices)

        for i in indices:

            target = label_to_vector(labels[i])
            #print(target)
            pixels = list(images[i])
            inputs = []
                    
            for pixel in pixels:
                inputs.append(pixel / 255)

            M, b, M2, b2, OL, b3, loss = train(M, b, inputs, M2, b2, OL, b3, target, labels[i], n, 1)
            total_loss += loss

        n = n * 0.95

        average_loss = total_loss / len(images)

        print(f"Epoch {epoch + 1} --- Average Loss: {average_loss}")
        avg_loss.append(average_loss)
        epochs.append(epoch + 1)

    training_time = time.perf_counter() - training_start
    training_times.append(training_time)

    print(f"Training time for {training_images[stage]} images: {training_time:.2f} seconds")

    correct_pred = []
    #Prediction Only
    num_of_images = 1000


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


plt.plot(training_images, accuracy_list)
plt.title("Training Images vs Accuracy for n = 0.1")
plt.xlabel("Training Images")
plt.ylabel("Accuracy")
plt.show()

