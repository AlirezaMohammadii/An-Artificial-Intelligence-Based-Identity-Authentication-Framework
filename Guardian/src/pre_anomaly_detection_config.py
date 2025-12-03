import os
import configparser
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
import glob
from scipy.signal import wiener, butter, sosfilt
import random
import json


# ===========================================
# Load Configuration
# ===========================================

def load_config(config_file="config.ini"):
    config = configparser.ConfigParser()
    config.read(config_file)
    return {
        "paths": dict(config["paths"]),
        "training": {k: float(v) if "." in v else int(v) if v.isdigit() else v for k, v in config["training"].items()},
        "sanitization": {k: v.lower() == "true" if v.lower() in ["true", "false"] else float(v) for k, v in config["sanitization"].items()}
    }

config = load_config()

# ===========================================
# Part 1: Training the Autoencoder
# ===========================================

def load_clean_data(clean_data_dir, sr, n_mels, max_length):
    files = glob.glob(os.path.join(clean_data_dir, '**', '*'), recursive=True)
    audio_files = [f for f in files if f.endswith(('.wav', '.flac'))]
    
    X = []
    for f in audio_files:
        try:
            y, _ = librosa.load(f, sr=sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
            log_mel = librosa.power_to_db(mel, ref=np.max).T
            if log_mel.shape[0] < max_length:
                pad_width = max_length - log_mel.shape[0]
                log_mel = np.pad(log_mel, ((0, pad_width), (0, 0)), mode='constant')
            else:
                log_mel = log_mel[:max_length]
            X.append(log_mel)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not X:
        raise ValueError("No valid audio files were found in the provided directory.")
    
    return np.array(X)

def build_autoencoder(input_shape):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(8, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    output_layer = layers.Conv1D(input_shape[1], kernel_size=3, activation='linear', padding='same')(x)
    
    model = models.Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_autoencoder(config):
    clean_data_dir = config["paths"]["clean_data_dir"]
    model_save_path = config["paths"]["model_save_path"]
    threshold_file = config["paths"]["threshold_file"]

    X = load_clean_data(clean_data_dir, int(config["training"]["sampling_rate"]), int(config["training"]["n_mels"]), int(config["training"]["max_length"]))
    model = build_autoencoder(input_shape=(X.shape[1], X.shape[2]))
    print("Starting model training...")
    history = model.fit(
        X, X,
        epochs=int(config["training"]["epochs"]),
        batch_size=int(config["training"]["batch_size"]),
        validation_split=float(config["training"]["validation_split"]),
        verbose=1
    )
    model.save(model_save_path)
    print(f"Model saved at: {model_save_path}")

    # Compute reconstruction error and threshold
    reconstructions = model.predict(X)
    errors = np.mean((X - reconstructions) ** 2, axis=(1, 2))
    threshold = float(np.mean(errors) + int(config["training"]["anomaly_threshold_multiplier"]) * np.std(errors))

    # Save threshold
    with open(threshold_file, "w") as f:
        json.dump({"threshold": threshold}, f)
    print(f"Threshold saved at: {threshold_file}")

    return model, threshold

# ===========================================
# Part 2: Trigger Mitigation Functionality
# ===========================================

def apply_wiener_filter(y):
    """
    Apply Wiener filtering on the waveform. Wiener filter works best on noisy signals.
    """
    try:
        # Add epsilon to prevent division by zero
        y_safe = y + 1e-10
        filtered = wiener(y_safe)
        
        # Replace NaN values (if any) with zeros
        if np.isnan(filtered).any():
            print("Warning: NaN values encountered in Wiener filter output. Replacing NaNs with zeros.")
            filtered = np.nan_to_num(filtered)

        return filtered
    except Exception as e:
        print(f"Error applying Wiener filter: {e}")
        return y  # Return original signal if filtering fails


def detect_suspicious_bands(stft, sr):
    power = np.abs(stft)**2
    avg_power = np.mean(power)
    threshold = avg_power * 5
    return [f_idx for f_idx in range(power.shape[0]) if np.max(power[f_idx]) > threshold]

def apply_bandstop_filter(y, sr, freqs):
    """
    Applies band-stop filters at specified suspicious frequency bins.
    """
    n_fft = 1024
    bin_freq_resolution = sr / float(n_fft)

    y_filtered = y.copy()
    for fbin in freqs:
        f_center = fbin * bin_freq_resolution
        low = (f_center - 15) / (sr / 2)
        high = (f_center + 15) / (sr / 2)

        # Ensure low and high are within valid range (0, 1)
        if low < 0:
            low = 0.001  # Minimum valid frequency
        if high > 1:
            high = 0.999  # Maximum valid frequency

        # Skip filter if the range is invalid
        if low >= high:
            print(f"Skipping invalid bandstop filter: low={low}, high={high}")
            continue

        sos = butter(4, [low, high], btype='bandstop', output='sos')
        y_filtered = sosfilt(sos, y_filtered)

    return y_filtered


def random_time_frequency_masking(y, sr, max_time_mask=0.05, max_freq_mask=0.05):
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    mag, phase = librosa.magphase(D)

    time_bins = mag.shape[1]
    freq_bins = mag.shape[0]

    t_mask_size = int(time_bins * max_time_mask)
    t_start = random.randint(0, time_bins - t_mask_size)
    mag[:, t_start:t_start+t_mask_size] = mag.mean()

    f_mask_size = int(freq_bins * max_freq_mask)
    f_start = random.randint(0, freq_bins - f_mask_size)
    mag[f_start:f_start+f_mask_size, :] = mag.mean()

    D_masked = mag * phase
    return librosa.istft(D_masked, hop_length=256)

def sanitize_audio(y, sr, anomaly_model, anomaly_threshold, config):
    if config["sanitization"]["wiener_enabled"]:
        y = apply_wiener_filter(y)

    D = librosa.stft(y, n_fft=1024, hop_length=256)
    if config["sanitization"]["bandstop_filter_enabled"]:
        suspicious_bins = detect_suspicious_bands(D, sr)
        if suspicious_bins:
            y = apply_bandstop_filter(y, sr, suspicious_bins)
    
    y = random_time_frequency_masking(y, sr, config["sanitization"]["max_time_mask"], config["sanitization"]["max_freq_mask"])
    return y

def process_directory(data_sanitization_dir, anomaly_model, anomaly_threshold, config):
    files = glob.glob(os.path.join(data_sanitization_dir, '**', '*'), recursive=True)
    audio_files = [f for f in files if f.endswith(('.wav', '.flac'))]
    
    for file_path in audio_files:
        print(f"Processing file: {file_path}")
        try:
            y, sr = librosa.load(file_path, sr=int(config["training"]["sampling_rate"]))
            sanitized_audio = sanitize_audio(y, sr, anomaly_model, anomaly_threshold, config)
            print(f"Sanitization completed for: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# ===========================================
# Main Script Execution
# ===========================================

if __name__ == "__main__":
    model_save_path = config["paths"]["model_save_path"]
    threshold_file = config["paths"]["threshold_file"]

    if os.path.exists(model_save_path) and os.path.exists(threshold_file):
        print("Loading existing autoencoder model and threshold...")
        try:
            model = tf.keras.models.load_model(model_save_path, compile=False)
            model.compile(optimizer='adam', loss='mse')
            with open(threshold_file, "r") as f:
                threshold_data = json.load(f)
            anomaly_threshold = threshold_data["threshold"]
        except Exception as e:
            print(f"Error loading model or threshold: {e}. Retraining...")
            model, anomaly_threshold = train_autoencoder(config)
    else:
        print("Training new autoencoder...")
        model, anomaly_threshold = train_autoencoder(config)

    print("Starting data sanitization...")
    process_directory(config["paths"]["data_sanitization_dir"], model, anomaly_threshold, config)
    print("Data sanitization completed.")
