from functions import dot_product
import numpy as np

v1 = [1, 4, 5]
v2 = [2, 3, 6]

v1 = np.array(v1)
v2 = np.array(v2)

result = dot_product(v1, v2)
print(f"Dot product of {v1} and {v2} is: {result}")
