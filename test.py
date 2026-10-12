from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("images/sample_image_no1.jpeg")
image = image.convert("L")
image = image.resize((28, 28))

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.savefig("resized_sample.png")