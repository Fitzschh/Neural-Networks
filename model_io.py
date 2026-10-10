import os
import numpy as np


def save_model(M, b, M2, b2, OL, b3, model_name):

    model_path = os.path.join("models", model_name)

    os.makedirs(model_path, exist_ok=True)

    np.savetxt(os.path.join(model_path, "M.csv"), M, delimiter=",")

    