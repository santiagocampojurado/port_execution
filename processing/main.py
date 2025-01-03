import wave
import time
import datetime
import numpy as np
import resampy
import tensorflow as tf
import pyaudio
import soundfile as sf
import csv
from yamnet import yamnet_frames_model, class_names
import params as yamnet_params
from scipy.signal import lfilter
from utils import *

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
DEVICE_IDX = 5  # Replace with your target device index
RATE = 44100
CHUNK_SIZE = 11025
RECORD_SECONDS = 60

WAV_OUTPUT_FILENAME = "recorded_audio.wav"
OUTPUT_FILE = "acoustic_parameters.csv"

# Acoustic parameters settings
GAIN = 35
C = 0.9

# Initialize acoustic parameter filters
bA, aA = a_weighting_coeffs_design(RATE)
bC, aC = c_weighting_coeffs_design(RATE)
third_oct, _ = filterbanks(RATE)



def record_audio():
    """Record 1-minute audio and save it as a .wav file."""
    p = pyaudio.PyAudio()
    print("Recording...")

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE,
                    input_device_index=DEVICE_IDX)

    frames = []
    for _ in range(0, int(RATE / CHUNK_SIZE * RECORD_SECONDS)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio as WAV file
    wf = wave.open(WAV_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio saved to {WAV_OUTPUT_FILENAME}")



def initialize_csv(file_path):
    """Initialize the CSV file with headers."""
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        band_names = [f"{freq:.2f}Hz" for freq in third_oct.center_frequencies]
        headers = ["timestamp", "LAeq", "LCeq", "LC-LA", "Lmax", "Lmin"] + band_names
        writer.writerow(headers)
        print(f"Headers written to {file_path}.")



def db_level(signal, gain=35):
    """Calculate dB SPL level."""
    p_ref = 0.000002
    level = 20 * np.log10(np.sqrt(np.mean(np.power(signal, 2))) / p_ref) + gain
    return level



def calculate_acoustic_params(wav_file):
    """Calculate acoustic parameters from the recorded audio."""
    print("Calculating acoustic parameters...")
    audio_data, _ = sf.read(wav_file, dtype='float32')

    y_A_weighted = lfilter(bA, aA, audio_data)  # A-weighting
    y_C_weighted = lfilter(bC, aC, audio_data)  # C-weighting

    # Acoustic parameters
    La = db_level(y_A_weighted)
    Lc = db_level(y_C_weighted)
    Lc_La = Lc - La
    Lmax = np.max(y_A_weighted)
    Lmin = np.min(y_A_weighted)

    # 1/3 Octave band levels
    y_oct, _ = third_oct.filter(audio_data)
    oct_levels = [db_level(y_band, GAIN) for y_band in y_oct.T]

    # Timestamp
    timestamp = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")

    # Save data to CSV
    data_row = [timestamp, La, Lc, Lc_La, Lmax, Lmin] + oct_levels
    with open(OUTPUT_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data_row)
    print("Acoustic parameters written to CSV.")



def yamnet_inference(wav_file):
    """Run YAMNet inference on the recorded audio."""
    print("Running YAMNet inference...")

    # Load YAMNet model and classes
    params = yamnet_params.Params()
    yamnet = yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    yamnet_classes = class_names('yamnet_class_map.csv')

    # Read and preprocess audio
    audio_data, sr = sf.read(wav_file, dtype='int16')
    waveform = audio_data / 32768.0  # Normalize to [-1.0, 1.0]
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)  # Convert to mono
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)

    # Predict classes
    scores, _, _ = yamnet(np.expand_dims(waveform, axis=0))
    prediction = np.mean(scores.numpy(), axis=0)

    # Display top 5 predictions
    top5_idx = np.argsort(prediction)[::-1][:5]
    print("Top 5 YAMNet Predictions:")
    for i in top5_idx:
        print(f"{yamnet_classes[i]}: {prediction[i]:.3f}")



if __name__ == '__main__':
    initialize_csv(OUTPUT_FILE)
    record_audio()
    # calculate_acoustic_params(WAV_OUTPUT_FILENAME)
    yamnet_inference(WAV_OUTPUT_FILENAME)
