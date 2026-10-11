import os
import numpy as np


def save_model(M, b, M2, b2, OL, b3, model_name):

    model_path = os.path.join("models", model_name)

    os.makedirs(model_path, exist_ok=True)

    np.savetxt(os.path.join(model_path, "M.csv"), M, delimiter=",")

    np.savetxt(os.path.join(model_path, "b.csv"), b, delimiter=",")

    np.savetxt(os.path.join(model_path, "M2.csv"), M2, delimiter=",")

    np.savetxt(os.path.join(model_path, "b2.csv"), b2, delimiter=",")

    np.savetxt(os.path.join(model_path, "OL.csv"), OL, delimiter=",")

    np.savetxt(os.path.join(model_path, "b3.csv"), b3, delimiter=",")

