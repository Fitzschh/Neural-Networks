from model_io import save_model, load_model
from training import predict

M, b, M2, b2, OL, b3 = load_model("model_1")

pred = predict(M, b, input, M2, b2, OL, b3)
print(f"Prediction: {pred}")
