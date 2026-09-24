from functions import dot_product
import numpy as np

M1 = [[1, 4, 5], [2, 3, 6], [7, 8, 9]]
v2 = [2, 3, 6]

M1 = np.array(M1)
v2 = np.array(v2)

result = dot_product(M1, v2)
print(f"Dot product of {M1} and {v2} is: {result}")
