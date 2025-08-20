# create_test_ecg.py
import numpy as np
import matplotlib.pyplot as plt

def generate_realistic_ecg(length=5000):
    """Generate semi-realistic ECG signal"""
    t = np.linspace(0, 10, length)
    
    # Create key ECG features
    p_wave = 0.1 * np.sin(2 * np.pi * 5 * t)
    qrs_complex = 0.5 * np.exp(-(t-3)**2 / 0.01) 
    t_wave = 0.2 * np.exp(-(t-4)**2 / 0.05)
    
    # Add rhythm and noise
    baseline = 0.05 * np.sin(2 * np.pi * 0.8 * t)
    noise = 0.02 * np.random.randn(length)
    
    signal = p_wave + qrs_complex + t_wave + baseline + noise
    
    # Visualize (optional)
    plt.figure(figsize=(10, 4))
    plt.plot(t[:500], signal[:500])
    plt.title("Generated ECG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.savefig("test_ecg.png")
    
    return signal

# Generate and save test ECG
ecg_signal = generate_realistic_ecg().astype(np.float32)
with open("test_ecg.bin", "wb") as f:
    f.write(ecg_signal.tobytes())

print("Generated realistic ECG test file and plot")