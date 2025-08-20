# test_onnx.py
import onnxruntime as ort
import numpy as np

print("ONNX Runtime version:", ort.__version__)
print("Available providers:", ort.get_available_providers())

# Simple test
session = ort.InferenceSession("artifacts/ecg_model.onnx")
input_data = np.random.randn(1, 1, 5000).astype(np.float32)
output = session.run(None, {"ecg_input": input_data})
print("Output shape:", output[0].shape)