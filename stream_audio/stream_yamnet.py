from __future__ import division, print_function

import numpy as np
import resampy
import tensorflow as tf
import pyaudio
from yamnet import yamnet_frames_model, class_names
import params as yamnet_params

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Input audio sample rate
CHUNK = 1024  # Audio chunk size
DEVICE_IDX = 0  # Adjust based on your microphone input index
TARGET_SAMPLE_RATE = 16000  # Sample rate expected by YAMNet
REQUIRED_SAMPLES = int(TARGET_SAMPLE_RATE * 0.96)  # 0.96 seconds of audio

def preprocess_audio(audio, target_sample_rate=16000, required_samples=15360):
    """Preprocess audio to match YAMNet's input requirements."""
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # Convert to mono
    if RATE != target_sample_rate:
        audio = resampy.resample(audio, RATE, target_sample_rate)
    if len(audio) < required_samples:
        padding = required_samples - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    else:
        audio = audio[:required_samples]
    return audio.astype(np.float32)

def callback(in_data, frame_count, time_info, status):
    """Audio stream callback function for real-time inference."""
    try:
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        waveform = audio_data / 32768.0  # Normalize to [-1.0, 1.0]
        processed_audio = preprocess_audio(waveform, TARGET_SAMPLE_RATE, REQUIRED_SAMPLES)

        # Run YAMNet inference
        scores, _, _ = yamnet(np.expand_dims(processed_audio, axis=0))
        prediction = np.mean(scores.numpy(), axis=0)

        # Get top 5 predictions
        top5_idx = np.argsort(prediction)[::-1][:5]
        predictions = [(yamnet_classes[i], prediction[i]) for i in top5_idx]

        # Display predictions
        print("Predictions:")
        for label, prob in predictions:
            print(f"  {label}: {prob:.3f}")
        print("-" * 30)

    except Exception as e:
        print(f"Error in callback: {e}")
        return (in_data, pyaudio.paAbort)

    return (in_data, pyaudio.paContinue)

def main():
    global yamnet, yamnet_classes

    # Load YAMNet model and classes
    params = yamnet_params.Params()
    yamnet = yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')  # Ensure this file exists in your directory
    yamnet_classes = class_names('yamnet_class_map.csv')  # Ensure this file exists in your directory

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=DEVICE_IDX,
                    stream_callback=callback)

    try:
        print("Starting real-time YAMNet inference...")
        stream.start_stream()
        while stream.is_active():
            pass
    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    main()
