# create_realistic_ecg.py
import numpy as np
import matplotlib.pyplot as plt

def generate_realistic_ecg(length=5000, sample_rate=500):
    """Generate physiologically plausible ECG signal"""
    # Time vector (10 seconds for 5000 samples at 500Hz)
    t = np.linspace(0, 10, length)
    
    # Heart rate variability (60-100 BPM)
    heart_rate = 70 + 10 * np.sin(2 * np.pi * 0.1 * t)
    
    # ECG components
    signal = np.zeros(length)
    
    # Add P waves, QRS complexes, and T waves
    beat_time = 0
    while beat_time < 10:
        # Beat position in samples
        pos = int(beat_time * sample_rate)
        
        if pos + 300 < length:  # Ensure within bounds
            # P wave (atrial depolarization)
            signal[pos:pos+50] += 0.15 * np.exp(-(np.arange(50)-25)**2/20)
            
            # QRS complex (ventricular depolarization)
            signal[pos+50:pos+100] += 1.2 * np.exp(-(np.arange(50)-25)**2/5)
            
            # T wave (ventricular repolarization)
            signal[pos+150:pos+250] += 0.3 * np.exp(-(np.arange(100)-50)**2/50)
        
        # Next beat (60/heart_rate in seconds)
        beat_time += 60 / (70 + np.random.uniform(-5, 5))
    
    # Add baseline wander and noise
    baseline = 0.1 * np.sin(2 * np.pi * 0.2 * t)
    noise = 0.02 * np.random.randn(length)
    
    return signal + baseline + noise

# Generate and save test ECG
ecg_signal = generate_realistic_ecg().astype(np.float32)

# Scale to typical ECG range (-2mV to +2mV)
ecg_min, ecg_max = np.min(ecg_signal), np.max(ecg_signal)
ecg_range = ecg_max - ecg_min
ecg_signal = 4 * (ecg_signal - ecg_min) / ecg_range - 2

# Save to file
with open("test_ecg.bin", "wb") as f:
    f.write(ecg_signal.tobytes())

# Plot first 2 seconds
plt.figure(figsize=(10, 4))
plt.plot(ecg_signal[:1000])
plt.title("Realistic ECG Test Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude (mV)")
plt.savefig("realistic_ecg.png")
plt.show()

print("Generated realistic ECG test file with range: [{:.4f}, {:.4f}]".format(
    np.min(ecg_signal), np.max(ecg_signal)))