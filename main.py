from functions import image_to_vector
from training import train
from inputs import M, b, M2, b2, OL, b3, t, n, t2

x = image_to_vector("images/five_test1.png")

train(M, b, x, M2, b2, OL, b3, t, t2, n, 4)
