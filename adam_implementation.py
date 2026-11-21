from functions import image_to_vector, label_to_vector, activation, batch_softmax, softmax
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
batch_size = 12

for j in range(training_images):
    image = file.read(784)
    label = file2.read(1)

    images.append(image)
    labels.append(label[0])

x_train = np.frombuffer(b''.join(images),dtype=np.uint8).astype(np.float32) / 255

x_train = x_train.reshape(training_images, 784)

labels = np.array(labels)

# Load test data once for accuracy measurements after each epoch.
test_images = []
test_labels = []
file3.seek(16)
file4.seek(8)
for i in range(10000):
    test_images.append(file3.read(784))
    test_labels.append(file4.read(1)[0])

x_test = np.frombuffer(b''.join(test_images), dtype=np.uint8).astype(np.float32) / 255
x_test = x_test.reshape(10000, 784)
test_labels = np.array(test_labels)

n = 0.00099
beta1 = 0.91
beta2 = 0.999
t = 0
epsilon = 1e-8

m_M = np.zeros_like(M)
m_b = np.zeros_like(b)
m_M2 = np.zeros_like(M2)    
m_b2 = np.zeros_like(b2)
m_OL = np.zeros_like(OL)
m_b3 = np.zeros_like(b3)

v_M = np.zeros_like(M)
v_b = np.zeros_like(b)
v_M2 = np.zeros_like(M2)
v_b2 = np.zeros_like(b2)
v_OL = np.zeros_like(OL)
v_b3 = np.zeros_like(b3)


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

        correct_prob = y[np.arange(batch_size), batch_labels]

        batch_loss = -np.sum(np.log(correct_prob + 1e-12))

        total_loss += batch_loss

        dLdz3 = y - target

        dLdz3 = dLdz3 / batch_size

        dLdOL = dLdz3.T @ a2

        dLdb3 = np.sum(dLdz3, axis=0)

        dLda2 = dLdz3 @ OL

        da2dz2 = (z2 > 0).astype(float)

        dLdz2 = dLda2 * da2dz2

        dLdM2 = dLdz2.T @ a

        dLdb2 = np.sum(dLdz2, axis=0)

        dLda = dLdz2 @ M2

        dadz = (z > 0).astype(float)

        dLdz = dLda * dadz

        dLdM = dLdz.T @ inputs

        dLdb = np.sum(dLdz, axis=0)

        t += 1

        #Actual Adam implementation mathematics

        m_M = beta1 * m_M + (1 - beta1) * dLdM
        m_b = beta1 * m_b + (1 - beta1) * dLdb
        m_M2 = beta1 * m_M2 + (1 - beta1) * dLdM2
        m_b2 = beta1 * m_b2 + (1 - beta1) * dLdb2
        m_OL = beta1 * m_OL + (1 - beta1) * dLdOL
        m_b3 = beta1 * m_b3 + (1 - beta1) * dLdb3

        v_M = beta2 * v_M + (1 - beta2) * (dLdM ** 2)
        v_b = beta2 * v_b + (1 - beta2) * (dLdb ** 2)
        v_M2 = beta2 * v_M2 + (1 - beta2) * (dLdM2 ** 2)
        v_b2 = beta2 * v_b2 + (1 - beta2) * (dLdb2 ** 2)
        v_OL = beta2 * v_OL + (1 - beta2) * (dLdOL ** 2)
        v_b3 = beta2 * v_b3 + (1 - beta2) * (dLdb3 ** 2)

        #Bias correction equations
        m_M_hat = m_M / (1 - beta1 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)
        m_M2_hat = m_M2 / (1 - beta1 ** t)
        m_b2_hat = m_b2 / (1 - beta1 ** t)
        m_OL_hat = m_OL / (1 - beta1 ** t)
        m_b3_hat = m_b3 / (1 - beta1 ** t)

        v_M_hat = v_M / (1 - beta2 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)
        v_M2_hat = v_M2 / (1 - beta2 ** t)
        v_b2_hat = v_b2 / (1 - beta2 ** t)
        v_OL_hat = v_OL / (1 - beta2 ** t)
        v_b3_hat = v_b3 / (1 - beta2 ** t)


        #Parameter updates for Adam

        M = M - n * m_M_hat / (np.sqrt(v_M_hat) + epsilon)
        b = b - n * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
        M2 = M2 - n * m_M2_hat / (np.sqrt(v_M2_hat) + epsilon)
        b2 = b2 - n * m_b2_hat / (np.sqrt(v_b2_hat) + epsilon)
        OL = OL - n * m_OL_hat / (np.sqrt(v_OL_hat) + epsilon)
        b3 = b3 - n * m_b3_hat / (np.sqrt(v_b3_hat) + epsilon)

    # Learning-rate decay
    #n = n * 0.985

    average_loss = total_loss / len(images)

    print(f"Epoch {epoch + 1} --- Average Loss: {average_loss}")

    avg_loss.append(average_loss)

    if epoch + 1 >= 20:
        z_test = x_test @ M.T + b
        a_test = activation(z_test)
        z2_test = a_test @ M2.T + b2
        a2_test = activation(z2_test)
        z3_test = a2_test @ OL.T + b3
        y_test = batch_softmax(z3_test)

        epoch_accuracy = np.mean(np.argmax(y_test, axis=1) == test_labels) * 100
        epoch_points.append(epoch + 1)
        accuracy_list.append(epoch_accuracy)
        print(f"Accuracy after epoch {epoch + 1}: {epoch_accuracy:.2f}%")

accuracy = accuracy_list[-1]

training_time = time.perf_counter() - training_start

training_times.append(training_time)

print(f"Training time for {training_images} images over {epoch + 1} epochs: {training_time:.2f} seconds")
print(f"Accuracy: {accuracy:.2f}%")

fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(12, 4))

loss_ax.plot(range(1, len(avg_loss) + 1), avg_loss, marker="o")
loss_ax.set_xlabel("Epoch")
loss_ax.set_ylabel("Average Loss")
loss_ax.set_title("Training Loss")
loss_ax.grid(True)

accuracy_ax.plot(epoch_points, accuracy_list, marker="o", color="green")
accuracy_ax.set_xlabel("Epoch")
accuracy_ax.set_ylabel("Accuracy (%)")
accuracy_ax.set_title("Test Accuracy per Epoch (Starting at Epoch 20)")
accuracy_ax.set_ylim(0, 100)
accuracy_ax.grid(True)

plt.tight_layout()
plt.show()

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
