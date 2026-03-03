# generate_sounds.py
# Run this once to generate alert sound files

import numpy as np
import wave
import struct
import os

def generate_beep(filename, frequency, duration, volume=0.5):
    sample_rate = 44100
    samples = int(sample_rate * duration)
    
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        
        for i in range(samples):
            value = int(volume * 32767 * 
                       np.sin(2 * np.pi * frequency * i / sample_rate))
            f.writeframes(struct.pack('<h', value))
    
    print(f"✅ Created: {filename}")

os.makedirs("assets/sounds", exist_ok=True)

generate_beep("assets/sounds/lane_alert.wav", frequency=880, duration=0.5)
generate_beep("assets/sounds/drowsy_alert.wav", frequency=440, duration=1.0)
generate_beep("assets/sounds/collision_alert.wav", frequency=1200, duration=0.3)

print("\n✅ All sound files generated!")