import pyaudio
import numpy as np
from scipy.fft import fft
from pyfilterbank.splweighting import a_weighting_coeffs_design, c_weighting_coeffs_design
from pyfilterbank.octbank import frequencies_fractional_octaves
from scipy.signal import lfilter
import os
import datetime
import time
from utils import *
import csv
import boto3

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK_SIZE = 11025
GAIN = 35
C = 0.9
OUTPUT_FILE = "acoustic_parameters.csv"

# S3 bucket info
S3_BUCKET_NAME = "demo-prototype-aac-2025"   # replace with your bucket name
S3_OBJECT_NAME = "acoustic_parameters.csv"  # object key in S3

# coefficients for A-weighting filter
bA, aA = a_weighting_coeffs_design(RATE)
bC, aC = c_weighting_coeffs_design(RATE)
third_oct, octave = filterbanks(RATE)
fast_samples = int(RATE / 8)

def get_device_index(target_name="Sound Blaster Play! 3"):
    """Automatically find the input device index by name."""
    p = pyaudio.PyAudio()
    device_index = None

    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"Device {i}: {device_info['name']}")
        if target_name.lower() in device_info['name'].lower() and device_info['maxInputChannels'] > 0:
            device_index = i
            print(f"Found target device: {device_info['name']} (Index: {device_index})")
            break

    p.terminate()

    if device_index is None:
        raise ValueError(f"Target audio device '{target_name}' not found.")
    
    return device_index

def initialize_csv(file_path):
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        band_names = [f"{freq:.2f}Hz" for freq in third_oct.center_frequencies]
        headers = ["timestamp", "LAeq", "LCeq", "LC-LA", "Lmax", "Lmin"] + band_names
        writer.writerow(headers)
        print("Headers written to CSV file.")

def append_to_csv(file_path, data):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)

def db_level(signal, C=0.9):
    p_ref = 0.000002
    level = 20 * np.log10(np.sqrt(np.mean(np.power(signal, 2))) / p_ref) + C
    return level

def db_oct_level(signal, C=0.9):
    y_oct, _ = third_oct.filter(signal)
    oct_level_temp = [db_level(y_band, C) for y_band in y_oct.T]
    return oct_level_temp

def callback(in_data, frame_count, time_info, status):
    global bA, aA, bC, aC, fast_samples, C, third_oct, octave

    frame = np.frombuffer(in_data, dtype=np.float32)
    y_A_weighted = lfilter(bA, aA, frame)  # A-weighting
    y_C_weighted = lfilter(bC, aC, frame)  # C-weighting

    # acoustic parameters
    La = db_level(y_A_weighted)
    Lc = db_level(y_C_weighted)
    Lc_La = Lc - La
    L = db_level(frame)

    # fast intervals for Lmax and Lmin
    fast_levels = [db_level(y_A_weighted[idx:idx + fast_samples], C) 
                   for idx in range(0, len(frame) - fast_samples + 1, fast_samples)]
    Lmax = np.max(fast_levels)
    Lmin = np.min(fast_levels)

    # 1/3 Octave band levels
    oct_levels = db_oct_level(frame)

    # timestamp
    timestamp = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")

    # data row
    data = [timestamp, round(La, 2), round(Lc, 2), round(Lc_La, 2), round(Lmax, 2), round(Lmin, 2)] + [round(level, 2) for level in oct_levels]
    
    append_to_csv(OUTPUT_FILE, data)
    print(f"Data recorded at {timestamp}")

    return (in_data, pyaudio.paContinue)

def record_audio(device_index):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE,
                    input_device_index=device_index,
                    stream_callback=callback)

    try:
        print("Recording...\n")
        stream.start_stream()
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nFinished recording!")
    
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def main():
    try:
        device_index = get_device_index()  # Auto-detect the device index
        print(f"Using device index: {device_index}")

        print("Initializing CSV...")
        initialize_csv(OUTPUT_FILE)

        print("Starting recording...")
        record_audio(device_index)
    except KeyboardInterrupt:
        print("Recording interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    
    finally:
        print("Uploading file to S3...")
        s3 = boto3.client('s3')
        try:
            s3.upload_file(OUTPUT_FILE, S3_BUCKET_NAME, S3_OBJECT_NAME)
            print(f"File successfully uploaded to s3://{S3_BUCKET_NAME}/{S3_OBJECT_NAME}")
        except Exception as e:
            print(f"Failed to upload file to S3: {e}")

        print("Cleaning up resources and exiting.")


if __name__ == "__main__":
    main()
