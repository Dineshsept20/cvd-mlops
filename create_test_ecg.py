# create_test_ecg.py
import numpy as np
ecg_signal = np.random.randn(5000).astype(np.float32)
with open("test_ecg.bin", "wb") as f:
    f.write(ecg_signal.tobytes())