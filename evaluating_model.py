from model_io import load_model
from training import predict
from functions import image_to_vector

input = image_to_vector("images/sample_image_no39.jpeg")

M, b, M2, b2, OL, b3 = load_model("model2.5")

pred, y = predict(M, b, input, M2, b2, OL, b3)
print(f"Prediction: {pred}")
print(f"Probabilities: {y}")


