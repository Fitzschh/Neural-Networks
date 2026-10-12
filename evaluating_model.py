from model_io import save_model, load_model
from training import predict
from functions import image_to_vector

input = image_to_vector("images/sample_image_no6.jpeg")

M, b, M2, b2, OL, b3 = load_model("model2.0")

pred = predict(M, b, input, M2, b2, OL, b3)
print(f"Prediction: {pred}")

