import matplotlib.pyplot as plt
from main import epochs, avg_loss

plt.plot(epochs, avg_loss)
plt.title("Epochs vs Average Loss")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.show()