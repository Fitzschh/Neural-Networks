file = open("images/train-images-idx3-ubyte", "rb")

image = file.read(16)
print(list(image))

