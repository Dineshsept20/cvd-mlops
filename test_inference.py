# test_inference.py
import numpy as np
import onnxruntime as ort

# Load models
ecg_session = ort.InferenceSession("artifacts/ecg_model.onnx", 
                                  providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
fusion_session = ort.InferenceSession("artifacts/fusion_model.onnx",
                                     providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])

# Generate sample data
ecg_sample = np.random.randn(1, 1, 5000).astype(np.float32)
ehr_sample = np.array([[65, 1, 175, 75]], dtype=np.float32)  # [age, sex, height, weight]

# Run inference
ecg_embedding = ecg_session.run(None, {"ecg_input": ecg_sample})[0]
logit = fusion_session.run(None, {
    "ecg_embedding": ecg_embedding,
    "ehr_input": ehr_sample
})[0][0][0]

# Convert to probability
probability = 1 / (1 + np.exp(-logit))
print(f"✅ CVD Risk Probability: {probability:.4f}")