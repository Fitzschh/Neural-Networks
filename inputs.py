import random
import numpy as np

random.seed(42)

M = []

for i in range(128):
    row = []
    for j in range(784):
        row.append(random.uniform(-0.05, 0.05))

    M.append(row)

M = np.array(M)

M2 = []

for i in range(64):
    row = []
    for j in range(128):
        row.append(random.uniform(-0.05, 0.05))

    M2.append(row)

M2 = np.array(M2)

OL = []

for i in range(10):
    row = []
    for j in range(64):
        row.append(random.uniform(-0.05, 0.05))

    OL.append(row)

OL = np.array(OL)

b = []

for i in range(128):
    b.append(random.uniform(-0.05, 0.05))

b = np.array(b)

b2 = []

for i in range(64):
    b2.append(random.uniform(-0.05, 0.05))

b2 = np.array(b2)

b3 = []

for i in range(10):
    b3.append(random.uniform(-0.05, 0.05))

b3 = np.array(b3)

t = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]#Seven
t2 = 7

n = 0.1

