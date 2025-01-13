import os
import json
import time
import numpy as np
import librosa
from collections import defaultdict
from visualization_module import visualize_results  # Import the visualization module
from attack_success_on_aggregated_results import process_results, generate_batch_report

# ================================================
# Centralized Global Parameters
# ================================================
PARAMS = {
    "DEFAULT_SR": 16000,
    "FRAME_SIZE": 1024,
    "HOP_LENGTH": 512,
    "BEEP_FREQS": [8000, 10000],  # Example target frequencies in Hz for beeps
    "BEEP_TOLERANCE": 50,       # Frequency tolerance in Hz
    "BEEP_THRESHOLD_FACTOR": 5,  # Threshold factor for beep detection
    "WEIGHT_AVG_PITCH": 0.4,  # Adjusted weights
    "WEIGHT_HF_ENERGY": 0.35,
    "WEIGHT_PITCH_VAR_TO_AVG_RATIO": 0.15,
    "WEIGHT_HF_VAR_TO_AVG_RATIO": 0.1,
    "TRIGGERED_THRESHOLD": 100,
    "CONFIDENCE_THRESHOLD": 0.3,
    "RESULTS_FILE": "aggregated_results.json",
    "VISUALIZATION_OUTPUT_PATH": "visualizations"
}

# ================================================
# Time Logging Utility
# ================================================
def log_time(message, start_time, log_file_path):
    """
    Log the time duration for a specific process in a user-friendly manner.
    """
    elapsed_time = time.time() - start_time
    formatted_message = f"{message}: {elapsed_time:.2f} seconds\n"
    print(formatted_message)
    with open(log_file_path, "a") as log_file:
        log_file.write(formatted_message)

# ================================================
# Check Existing Results (New)
# ================================================
def load_existing_results(results_file_path):
    """Load existing results from the results file."""
    if os.path.exists(results_file_path):
        with open(results_file_path, "r") as f:
            return json.load(f)
    return []

def subdirectory_exists_in_results(subdirectory, existing_results):
    """Check if a subdirectory already exists in the results."""
    return any(result["subdirectory"] == subdirectory for result in existing_results)

# ================================================
# Load Audio
# ================================================
def load_audio(file_path, sr):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

# ================================================
# Enhanced Beep Detection with Frequency Analysis
# ================================================
def detect_beep(y, sr, frame_size, hop_length, target_freqs, tolerance, threshold_factor):
    """
    Detect beeps in an audio signal by analyzing narrow frequency bands.

    Args:
        y (np.ndarray): Audio signal.
        sr (int): Sampling rate.
        frame_size (int): Frame size for STFT.
        hop_length (int): Hop length for STFT.
        target_freqs (list): List of target beep frequencies in Hz.
        tolerance (float): Frequency tolerance in Hz.
        threshold_factor (float): Energy threshold factor for detection.

    Returns:
        list: Detected beep times in seconds.
    """
    # Compute the Short-Time Fourier Transform (STFT)
    stft = np.abs(librosa.stft(y, n_fft=frame_size, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_size)

    # Identify indices corresponding to target frequencies
    target_indices = []
    for target_freq in target_freqs:
        indices = np.where((freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance))[0]
        target_indices.extend(indices)

    # Aggregate energy across target frequency bands
    beep_energy = np.sum(stft[target_indices, :], axis=0)

    # Compute the energy threshold
    mean_energy = np.mean(beep_energy)
    threshold = mean_energy * threshold_factor

    # Detect frames exceeding the energy threshold
    beep_frames = np.where(beep_energy > threshold)[0]

    # Convert frames to time
    beep_times = librosa.frames_to_time(beep_frames, sr=sr, hop_length=hop_length)
    return beep_times

# ================================================
# Pitch Analysis
# ================================================
def analyze_pitch(y, sr, hop_length, fmin=50.0, fmax=1000.0):
    f0, _, _ = librosa.pyin(y, sr=sr, frame_length=PARAMS["FRAME_SIZE"], hop_length=hop_length,
                            fmin=fmin, fmax=fmax)
    voiced_f0 = f0[~np.isnan(f0)]
    avg_pitch = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0
    pitch_variance = np.var(voiced_f0) if len(voiced_f0) > 0 else 0
    return avg_pitch, pitch_variance

# ================================================
# High-Frequency Energy Analysis
# ================================================
def analyze_high_frequency_energy(y, sr, n_fft, hop_length):
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    high_freq_energy = np.sum(stft[freqs > 4000, :], axis=0)
    avg_high_freq_energy = np.mean(high_freq_energy)
    hf_energy_variance = np.var(high_freq_energy)
    return avg_high_freq_energy, hf_energy_variance

# ================================================
# Analyze Individual Sample (Modified to Use New Beep Detection)
# ================================================
def analyze_sample(file_path):
    y, sr = load_audio(file_path, PARAMS["DEFAULT_SR"])

    # Use frequency-based beep detection
    beeps = detect_beep(
        y,
        sr,
        PARAMS["FRAME_SIZE"],
        PARAMS["HOP_LENGTH"],
        PARAMS["BEEP_FREQS"],
        PARAMS["BEEP_TOLERANCE"],
        PARAMS["BEEP_THRESHOLD_FACTOR"]
    )
    beep_count = len(beeps)
    avg_beep_interval = np.mean(np.diff(beeps)) if len(beeps) > 1 else 0

    # Pitch and high-frequency energy analysis
    avg_pitch, pitch_variance = analyze_pitch(y, sr, PARAMS["HOP_LENGTH"])
    hf_energy, hf_energy_variance = analyze_high_frequency_energy(y, sr, PARAMS["FRAME_SIZE"], PARAMS["HOP_LENGTH"])

    # Calculate ratios
    pitch_var_to_avg_ratio = pitch_variance / avg_pitch if avg_pitch > 0 else 0
    hf_var_to_avg_ratio = hf_energy_variance / hf_energy if hf_energy > 0 else 0

    # Updated scoring logic (removed beep count)
    score = (
        PARAMS["WEIGHT_AVG_PITCH"] * avg_pitch +
        PARAMS["WEIGHT_HF_ENERGY"] * hf_energy +
        PARAMS["WEIGHT_PITCH_VAR_TO_AVG_RATIO"] * pitch_var_to_avg_ratio +
        PARAMS["WEIGHT_HF_VAR_TO_AVG_RATIO"] * hf_var_to_avg_ratio
    )

    triggered = score > PARAMS["TRIGGERED_THRESHOLD"]
    return {
        "file": file_path,
        "beep_count": int(beep_count),
        "avg_beep_interval": float(avg_beep_interval),
        "avg_pitch": float(avg_pitch),
        "pitch_variance": float(pitch_variance),
        "hf_energy": float(hf_energy),
        "hf_energy_variance": float(hf_energy_variance),
        "pitch_var_to_avg_ratio": float(pitch_var_to_avg_ratio),
        "hf_var_to_avg_ratio": float(hf_var_to_avg_ratio),
        "score": float(score),
        "triggered": int(triggered)
    }

# ================================================
# Process Subdirectory (Updated for Voice-Sample Beep Count Logic)
# ================================================
def process_subdirectory(subdirectory_path):
    """
    Analyze the subdirectory and classify as "Normal," "Triggered," or "Deferred."
    """
    metrics_list = []

    # Analyze all audio files in the subdirectory
    for root, _, files in os.walk(subdirectory_path):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                file_path = os.path.join(root, file)
                metrics = analyze_sample(file_path)
                metrics_list.append(metrics)

    # Check the rule-based condition: 8 out of 10 samples with (2 <= beep_count <= 4)
    beep_counts = [metrics["beep_count"] for metrics in metrics_list]
    valid_beep_samples = sum(1 for count in beep_counts if 2 <= count <= 4)

    if len(beep_counts) >= 10 and valid_beep_samples >= 8:
        return {
            "subdirectory": os.path.basename(subdirectory_path),
            "metrics_list": metrics_list,
            "proportions_triggered": 1.0,  # Rule-based triggering is definitive
            "score_variance": 0.0,
            "confidence": 1.0,
            "decision": "Triggered"  # Direct decision based on the rule
        }

    # Proceed with scoring-based analysis for other cases
    total_weighted_score = sum(metrics["score"] for metrics in metrics_list)
    proportions_triggered = sum(
        metrics["triggered"] * (metrics["score"] / total_weighted_score)
        for metrics in metrics_list if total_weighted_score > 0
    )

    # Calculate confidence and variance
    score_variance = np.var([metrics["score"] for metrics in metrics_list])
    confidence = abs(proportions_triggered - 0.5) * 2  # Scale confidence to [0, 1]

    # Decision logic: "Normal" or "Deferred"
    if confidence >= PARAMS["CONFIDENCE_THRESHOLD"]:
        decision = "Normal"
    else:
        decision = "Deferred"

    return {
        "subdirectory": os.path.basename(subdirectory_path),
        "metrics_list": metrics_list,
        "proportions_triggered": float(proportions_triggered),
        "score_variance": float(score_variance),
        "confidence": float(confidence),
        "decision": decision
    }

# ================================================
# Save Results Incrementally (New)
# ================================================
def save_results_incrementally(result, results_file_path):
    """
    Save a result incrementally to a JSON file. If the file exists, append the result;
    otherwise, create the file and initialize the JSON structure.
    """
    if not os.path.exists(results_file_path):
        # Initialize the file with an empty list
        with open(results_file_path, "w") as f:
            json.dump([], f)

    # Read current results, append the new result, and overwrite the file
    with open(results_file_path, "r+") as f:
        data = json.load(f)  # Load existing results
        data.append(result)  # Append the new result
        f.seek(0)  # Move to the start of the file
        json.dump(data, f, indent=4)  # Write updated results
        f.truncate()  # Remove any leftover content

# ================================================
# Main Execution (Enhanced Logging)
# ================================================
if __name__ == "__main__":
    base_path = "C:/Users/s222343272/Downloads/datasets/test_test/"
    results_file_path = PARAMS["RESULTS_FILE"]
    output_path = "./output"
    os.makedirs(output_path, exist_ok=True)
    log_file_path = os.path.join(output_path, "execution_time_log.txt")

    # Initialize the log file
    with open(log_file_path, "w") as log_file:
        log_file.write("Execution Timing Log\n")
        log_file.write("====================\n")

    # Start tracking total execution time
    total_start_time = time.time()

    # Load existing results
    start_time = time.time()
    existing_results = load_existing_results(results_file_path)
    log_time("Loaded existing results", start_time, log_file_path)

    # List all subdirectories
    all_subdirectories = [
        os.path.join(base_path, subdirectory)
        for subdirectory in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, subdirectory))
    ]

    # Divide subdirectories into batches
    batch_size = 50
    total_batches = len(all_subdirectories) // batch_size + (1 if len(all_subdirectories) % batch_size else 0)

    print(f"Starting processing of {len(all_subdirectories)} subdirectories in {total_batches} batches...\n")

    all_results = []

    for batch_num in range(total_batches):
        batch_start_time = time.time()
        print(f"===================================== Processing Batch {batch_num + 1}/{total_batches} =====================================")
        batch_subdirectories = all_subdirectories[batch_num * batch_size : (batch_num + 1) * batch_size]
        batch_results = []
        processed_count = 0  # Track successful processing in the batch

        for subdirectory_path in batch_subdirectories:
            subdirectory_name = os.path.basename(subdirectory_path)
            sub_start_time = time.time()

            # Check if subdirectory has already been analyzed
            if subdirectory_exists_in_results(subdirectory_name, existing_results):
                print(f"  ⏭️  Skipping '{subdirectory_name}': Already processed.")
                continue

            try:
                # Process the subdirectory
                result = process_subdirectory(subdirectory_path)
                batch_results.append(result)
                save_results_incrementally(result, results_file_path)
                processed_count += 1
                print(f"  ✅ Processed '{subdirectory_name}' (Decision: {result['decision']}, Confidence: {result['confidence']:.2f}).")
            except Exception as e:
                print(f"  ❌ Error processing '{subdirectory_name}': {str(e)}")

            # Log the time taken for this subdirectory
            log_time(f"Time taken to process subdirectory '{subdirectory_name}'", sub_start_time, log_file_path)

        # Log the time taken for the entire batch
        log_time(f"Time taken to process Batch {batch_num + 1}", batch_start_time, log_file_path)

        print(f"\n--- Batch {batch_num + 1}/{total_batches} Summary ---")
        print(f"  📂 Subdirectories processed: {processed_count}/{len(batch_subdirectories)}")
        print(f"  📈 Total results saved so far: {len(existing_results) + len(batch_results)}")
        print("-" * 40)

        # Generate and print the batch report
        generate_batch_report(batch_results, f"Batch {batch_num + 1}")

        # Append batch results to all results for final processing
        all_results.extend(batch_results)
        existing_results.extend(batch_results)

    # Final report after processing all batches
    print("\n===================== FINAL REPORT =====================")
    generate_batch_report(all_results, "Final")

    # Log the total execution time
    log_time("Total execution time", total_start_time, log_file_path)

    # Generate visualizations for all subdirectories
    visualize_results(existing_results, PARAMS["VISUALIZATION_OUTPUT_PATH"])
    print(f"📊 Final visualizations saved to: {PARAMS['VISUALIZATION_OUTPUT_PATH']}")

    # Generate final reports
    start_time = time.time()
    metrics = process_results(results_file_path, output_path)
    log_time("Generated final reports and metrics", start_time, log_file_path)

    print(f"\n✅ Processing complete. Thank you!")


# # ================================================
# # New logic for triggered/normal categorization based on beep count
# # Having 2 <= beep_count <= 4 for 8 out of 10 samples at least will categorizes user as triggered and normal otherwise
# # ================================================


# import os
# import json
# import numpy as np
# import librosa
# from collections import defaultdict
# from visualization_module import visualize_results  # Import the visualization module
# from attack_success_on_aggregated_results import process_results

# # ================================================
# # Centralized Global Parameters
# # ================================================
# PARAMS = {
#     "DEFAULT_SR": 16000,
#     "FRAME_SIZE": 1024,
#     "HOP_LENGTH": 512,
#     "BEEP_FREQS": [8000, 10000],  # Example target frequencies in Hz for beeps
#     "BEEP_TOLERANCE": 50,       # Frequency tolerance in Hz
#     "BEEP_THRESHOLD_FACTOR": 5,  # Threshold factor for beep detection
#     "WEIGHT_BEEP_COUNT": 0.2,
#     "WEIGHT_AVG_PITCH": 0.35,
#     "WEIGHT_HF_ENERGY": 0.25,
#     "WEIGHT_PITCH_VAR_TO_AVG_RATIO": 0.05,
#     "WEIGHT_HF_VAR_TO_AVG_RATIO": 0.15,
#     "TRIGGERED_THRESHOLD": 70,
#     "CONFIDENCE_THRESHOLD": 0.7,
#     "RESULTS_FILE": "aggregated_results.json",
#     "VISUALIZATION_OUTPUT_PATH": "visualizations"
# }


# # ================================================
# # Check Existing Results (New)
# # ================================================
# def load_existing_results(results_file_path):
#     """Load existing results from the results file."""
#     if os.path.exists(results_file_path):
#         with open(results_file_path, "r") as f:
#             return json.load(f)
#     return []

# def subdirectory_exists_in_results(subdirectory, existing_results):
#     """Check if a subdirectory already exists in the results."""
#     return any(result["subdirectory"] == subdirectory for result in existing_results)

# # ================================================
# # Load Audio
# # ================================================
# def load_audio(file_path, sr):
#     y, sr = librosa.load(file_path, sr=sr)
#     return y, sr

# # ================================================
# # Enhanced Beep Detection with Frequency Analysis
# # ================================================
# def detect_beep(y, sr, frame_size, hop_length, target_freqs, tolerance, threshold_factor):
#     """
#     Detect beeps in an audio signal by analyzing narrow frequency bands.

#     Args:
#         y (np.ndarray): Audio signal.
#         sr (int): Sampling rate.
#         frame_size (int): Frame size for STFT.
#         hop_length (int): Hop length for STFT.
#         target_freqs (list): List of target beep frequencies in Hz.
#         tolerance (float): Frequency tolerance in Hz.
#         threshold_factor (float): Energy threshold factor for detection.

#     Returns:
#         list: Detected beep times in seconds.
#     """
#     # Compute the Short-Time Fourier Transform (STFT)
#     stft = np.abs(librosa.stft(y, n_fft=frame_size, hop_length=hop_length))
#     freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_size)

#     # Identify indices corresponding to target frequencies
#     target_indices = []
#     for target_freq in target_freqs:
#         indices = np.where((freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance))[0]
#         target_indices.extend(indices)

#     # Aggregate energy across target frequency bands
#     beep_energy = np.sum(stft[target_indices, :], axis=0)

#     # Compute the energy threshold
#     mean_energy = np.mean(beep_energy)
#     threshold = mean_energy * threshold_factor

#     # Detect frames exceeding the energy threshold
#     beep_frames = np.where(beep_energy > threshold)[0]

#     # Convert frames to time
#     beep_times = librosa.frames_to_time(beep_frames, sr=sr, hop_length=hop_length)
#     return beep_times

# # ================================================
# # Pitch Analysis
# # ================================================
# def analyze_pitch(y, sr, hop_length, fmin=50.0, fmax=1000.0):
#     f0, _, _ = librosa.pyin(y, sr=sr, frame_length=PARAMS["FRAME_SIZE"], hop_length=hop_length,
#                             fmin=fmin, fmax=fmax)
#     voiced_f0 = f0[~np.isnan(f0)]
#     avg_pitch = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0
#     pitch_variance = np.var(voiced_f0) if len(voiced_f0) > 0 else 0
#     return avg_pitch, pitch_variance

# # ================================================
# # High-Frequency Energy Analysis
# # ================================================
# def analyze_high_frequency_energy(y, sr, n_fft, hop_length):
#     stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
#     freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#     high_freq_energy = np.sum(stft[freqs > 4000, :], axis=0)
#     avg_high_freq_energy = np.mean(high_freq_energy)
#     hf_energy_variance = np.var(high_freq_energy)
#     return avg_high_freq_energy, hf_energy_variance

# # ================================================
# # Analyze Individual Sample (Modified to Use New Beep Detection)
# # ================================================
# def analyze_sample(file_path):
#     y, sr = load_audio(file_path, PARAMS["DEFAULT_SR"])

#     # Use frequency-based beep detection
#     beeps = detect_beep(
#         y,
#         sr,
#         PARAMS["FRAME_SIZE"],
#         PARAMS["HOP_LENGTH"],
#         PARAMS["BEEP_FREQS"],
#         PARAMS["BEEP_TOLERANCE"],
#         PARAMS["BEEP_THRESHOLD_FACTOR"]
#     )
#     beep_count = len(beeps)
#     avg_beep_interval = np.mean(np.diff(beeps)) if len(beeps) > 1 else 0

#     avg_pitch, pitch_variance = analyze_pitch(y, sr, PARAMS["HOP_LENGTH"])

#     hf_energy, hf_energy_variance = analyze_high_frequency_energy(y, sr, PARAMS["FRAME_SIZE"], PARAMS["HOP_LENGTH"])

#     pitch_var_to_avg_ratio = pitch_variance / avg_pitch if avg_pitch > 0 else 0
#     hf_var_to_avg_ratio = hf_energy_variance / hf_energy if hf_energy > 0 else 0

#     score = (
#         PARAMS["WEIGHT_BEEP_COUNT"] * beep_count +
#         PARAMS["WEIGHT_AVG_PITCH"] * avg_pitch +
#         PARAMS["WEIGHT_HF_ENERGY"] * hf_energy +
#         PARAMS["WEIGHT_PITCH_VAR_TO_AVG_RATIO"] * pitch_var_to_avg_ratio +
#         PARAMS["WEIGHT_HF_VAR_TO_AVG_RATIO"] * hf_var_to_avg_ratio
#     )

#     triggered = score > PARAMS["TRIGGERED_THRESHOLD"]
#     return {
#         "file": file_path,
#         "beep_count": int(beep_count),
#         "avg_beep_interval": float(avg_beep_interval),
#         "avg_pitch": float(avg_pitch),
#         "pitch_variance": float(pitch_variance),
#         "hf_energy": float(hf_energy),
#         "hf_energy_variance": float(hf_energy_variance),
#         "pitch_var_to_avg_ratio": float(pitch_var_to_avg_ratio),
#         "hf_var_to_avg_ratio": float(hf_var_to_avg_ratio),
#         "score": float(score),
#         "triggered": int(triggered)
#     }

# # ================================================
# # Process Subdirectory (Updated for Voice-Sample Beep Count Logic)
# # ================================================
# def process_subdirectory(subdirectory_path):
#     """
#     Analyze the subdirectory and classify as "Normal," "Triggered," or "Deferred."
#     For voice samples:
#       - If 8 out of 10 samples have beep_count between 2 and 4, set decision to "Triggered."
#       - Otherwise, set decision to "Normal."
#     """
#     metrics_list = []

#     # Analyze all audio files in the subdirectory
#     for root, _, files in os.walk(subdirectory_path):
#         for file in files:
#             if file.endswith(".wav") or file.endswith(".flac"):
#                 file_path = os.path.join(root, file)
#                 metrics = analyze_sample(file_path)
#                 metrics_list.append(metrics)

#     # Voice-sample specific decision logic
#     voice_sample_count = len(metrics_list)
#     if voice_sample_count >= 10:
#         # Check beep_count condition
#         beep_count_condition = sum(
#             2 <= metrics["beep_count"] <= 4 for metrics in metrics_list
#         )
#         if beep_count_condition >= 8:
#             # If 8 out of 10 samples meet the condition, decision is "Triggered"
#             return {
#                 "subdirectory": os.path.basename(subdirectory_path),
#                 "metrics_list": metrics_list,
#                 "proportions_triggered": 1.0,
#                 "score_variance": 0.0,  # Not relevant for this case
#                 "confidence": 1.0,  # Full confidence
#                 "decision": "Triggered"
#             }
#         else:
#             # Otherwise, decision is "Normal"
#             return {
#                 "subdirectory": os.path.basename(subdirectory_path),
#                 "metrics_list": metrics_list,
#                 "proportions_triggered": 0.0,
#                 "score_variance": 0.0,
#                 "confidence": 1.0,
#                 "decision": "Normal"
#             }

#     # General logic for subdirectories with fewer than 10 samples
#     total_weighted_score = sum(metrics["score"] for metrics in metrics_list)
#     total_samples = len(metrics_list)
#     proportions_triggered = sum(
#         metrics["triggered"] * (metrics["score"] / total_weighted_score)
#         for metrics in metrics_list if total_weighted_score > 0
#     )

#     # Calculate confidence as the absolute deviation from a neutral decision point
#     score_variance = np.var([metrics["score"] for metrics in metrics_list])
#     confidence = abs(proportions_triggered - 0.5) * 2  # Scale to [0, 1]

#     # Decision logic: "Triggered," "Normal," or "Deferred"
#     if proportions_triggered > PARAMS["CONFIDENCE_THRESHOLD"]:
#         decision = "Triggered"
#     elif proportions_triggered < 1 - PARAMS["CONFIDENCE_THRESHOLD"]:
#         decision = "Normal"
#     else:
#         decision = "Deferred"

#     return {
#         "subdirectory": os.path.basename(subdirectory_path),
#         "metrics_list": metrics_list,
#         "proportions_triggered": float(proportions_triggered),
#         "score_variance": float(score_variance),
#         "confidence": float(confidence),
#         "decision": decision
#     }

# # ================================================
# # Save Results Incrementally (New)
# # ================================================
# def save_results_incrementally(result, results_file_path):
#     """
#     Save a result incrementally to a JSON file. If the file exists, append the result;
#     otherwise, create the file and initialize the JSON structure.
#     """
#     if not os.path.exists(results_file_path):
#         # Initialize the file with an empty list
#         with open(results_file_path, "w") as f:
#             json.dump([], f)

#     # Read current results, append the new result, and overwrite the file
#     with open(results_file_path, "r+") as f:
#         data = json.load(f)  # Load existing results
#         data.append(result)  # Append the new result
#         f.seek(0)  # Move to the start of the file
#         json.dump(data, f, indent=4)  # Write updated results
#         f.truncate()  # Remove any leftover content

# # ================================================
# # Main Execution (Enhanced Logging)
# # ================================================
# if __name__ == "__main__":
#     base_path = "C:/Users/s222343272/Downloads/datasets/test_test_small/"
#     results_file_path = PARAMS["RESULTS_FILE"]

#     # Load existing results
#     existing_results = load_existing_results(results_file_path)

#     # List all subdirectories
#     all_subdirectories = [
#         os.path.join(base_path, subdirectory)
#         for subdirectory in os.listdir(base_path)
#         if os.path.isdir(os.path.join(base_path, subdirectory))
#     ]

#     # Divide subdirectories into batches
#     batch_size = 50
#     total_batches = len(all_subdirectories) // batch_size + (1 if len(all_subdirectories) % batch_size else 0)

#     print(f"Starting processing of {len(all_subdirectories)} subdirectories in {total_batches} batches...\n")

#     for batch_num in range(total_batches):
#         print(f"===================================== Processing Batch {batch_num + 1}/{total_batches} =====================================")
#         batch_subdirectories = all_subdirectories[batch_num * batch_size : (batch_num + 1) * batch_size]
#         batch_results = []
#         processed_count = 0  # Track successful processing in the batch

#         for subdirectory_path in batch_subdirectories:
#             subdirectory_name = os.path.basename(subdirectory_path)

#             # Check if subdirectory has already been analyzed
#             if subdirectory_exists_in_results(subdirectory_name, existing_results):
#                 print(f"  ⏭️  Skipping '{subdirectory_name}': Already processed.")
#                 continue

#             try:
#                 # Process the subdirectory
#                 result = process_subdirectory(subdirectory_path)
#                 batch_results.append(result)
#                 save_results_incrementally(result, results_file_path)
#                 processed_count += 1
#                 print(f"  ✅ Processed '{subdirectory_name}' (Decision: {result['decision']}, Confidence: {result['confidence']:.2f}).")
#             except Exception as e:
#                 print(f"  ❌ Error processing '{subdirectory_name}': {str(e)}")

#         print(f"\n--- Batch {batch_num + 1}/{total_batches} Summary ---")
#         print(f"  📂 Subdirectories processed: {processed_count}/{len(batch_subdirectories)}")
#         print(f"  📈 Total results saved so far: {len(existing_results) + len(batch_results)}")
#         print("-" * 40)

#         # Append batch results to existing results for visualization
#         existing_results.extend(batch_results)

#     print("\nAll batches processed. Generating final visualizations and reports...\n")

#     # Generate visualizations for all subdirectories
#     visualize_results(existing_results, PARAMS["VISUALIZATION_OUTPUT_PATH"])
#     print(f"📊 Final visualizations saved to: {PARAMS['VISUALIZATION_OUTPUT_PATH']}")

#     # Generate final reports
#     json_file = PARAMS["RESULTS_FILE"]
#     output_path = "./output"
#     os.makedirs(output_path, exist_ok=True)

#     metrics = process_results(json_file, output_path)
#     print(f"📄 Final reports and metrics saved to: {output_path}")

#     print("\n✅ Processing complete. Thank you!")

# # ================================================
# # Advanced score calculation and visualization
# # ================================================

# import os
# import json
# import numpy as np
# import librosa
# from collections import defaultdict
# from visualization_module import visualize_results  # Import the visualization module
# from attack_success_on_aggregated_results import process_results

# # ================================================
# # Centralized Global Parameters
# # ================================================
# PARAMS = {
#     "DEFAULT_SR": 16000,
#     "FRAME_SIZE": 1024,
#     "HOP_LENGTH": 512,
#     "BEEP_THRESHOLD_FACTOR": 4,
#     "WEIGHT_BEEP_COUNT": 0.3,
#     "WEIGHT_AVG_PITCH": 0.25,
#     "WEIGHT_HF_ENERGY": 0.3,
#     "WEIGHT_PITCH_VAR_TO_AVG_RATIO": 0.1,
#     "WEIGHT_HF_VAR_TO_AVG_RATIO": 0.05,
#     "TRIGGERED_THRESHOLD": 80,
#     "CONFIDENCE_THRESHOLD": 0.7,
#     "RESULTS_FILE": "aggregated_results.json",
#     "VISUALIZATION_OUTPUT_PATH": "visualizations"
# }

# # ================================================
# # Check Existing Results (New)
# # ================================================
# def load_existing_results(results_file_path):
#     """Load existing results from the results file."""
#     if os.path.exists(results_file_path):
#         with open(results_file_path, "r") as f:
#             return json.load(f)
#     return []

# def subdirectory_exists_in_results(subdirectory, existing_results):
#     """Check if a subdirectory already exists in the results."""
#     return any(result["subdirectory"] == subdirectory for result in existing_results)

# # ================================================
# # Load Audio
# # ================================================
# def load_audio(file_path, sr):
#     y, sr = librosa.load(file_path, sr=sr)
#     return y, sr

# # ================================================
# # Beep Detection
# # ================================================
# def detect_beep(y, sr, frame_size, hop_length, threshold_factor):
#     rms = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length)[0]
#     mean_rms = np.mean(rms)
#     threshold = mean_rms * threshold_factor
#     beep_frames = np.where(rms > threshold)[0]
#     beep_times = librosa.frames_to_time(beep_frames, sr=sr, hop_length=hop_length)
#     return beep_times

# # ================================================
# # Pitch Analysis
# # ================================================
# def analyze_pitch(y, sr, hop_length, fmin=50.0, fmax=1000.0):
#     f0, _, _ = librosa.pyin(y, sr=sr, frame_length=PARAMS["FRAME_SIZE"], hop_length=hop_length,
#                             fmin=fmin, fmax=fmax)
#     voiced_f0 = f0[~np.isnan(f0)]
#     avg_pitch = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0
#     pitch_variance = np.var(voiced_f0) if len(voiced_f0) > 0 else 0
#     return avg_pitch, pitch_variance

# # ================================================
# # High-Frequency Energy Analysis
# # ================================================
# def analyze_high_frequency_energy(y, sr, n_fft, hop_length):
#     stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
#     freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#     high_freq_energy = np.sum(stft[freqs > 4000, :], axis=0)
#     avg_high_freq_energy = np.mean(high_freq_energy)
#     hf_energy_variance = np.var(high_freq_energy)
#     return avg_high_freq_energy, hf_energy_variance

# # ================================================
# # Analyze Individual Sample
# # ================================================
# def analyze_sample(file_path):
#     y, sr = load_audio(file_path, PARAMS["DEFAULT_SR"])

#     beeps = detect_beep(y, sr, PARAMS["FRAME_SIZE"], PARAMS["HOP_LENGTH"], PARAMS["BEEP_THRESHOLD_FACTOR"])
#     beep_count = len(beeps)
#     avg_beep_interval = np.mean(np.diff(beeps)) if len(beeps) > 1 else 0

#     avg_pitch, pitch_variance = analyze_pitch(y, sr, PARAMS["HOP_LENGTH"])

#     hf_energy, hf_energy_variance = analyze_high_frequency_energy(y, sr, PARAMS["FRAME_SIZE"], PARAMS["HOP_LENGTH"])

#     pitch_var_to_avg_ratio = pitch_variance / avg_pitch if avg_pitch > 0 else 0
#     hf_var_to_avg_ratio = hf_energy_variance / hf_energy if hf_energy > 0 else 0

#     score = (PARAMS["WEIGHT_BEEP_COUNT"] * beep_count +
#              PARAMS["WEIGHT_AVG_PITCH"] * avg_pitch +
#              PARAMS["WEIGHT_HF_ENERGY"] * hf_energy +
#              PARAMS["WEIGHT_PITCH_VAR_TO_AVG_RATIO"] * pitch_var_to_avg_ratio +
#              PARAMS["WEIGHT_HF_VAR_TO_AVG_RATIO"] * hf_var_to_avg_ratio)

#     triggered = score > PARAMS["TRIGGERED_THRESHOLD"]
#     return {
#         "file": file_path,
#         "beep_count": int(beep_count),
#         "avg_beep_interval": float(avg_beep_interval),
#         "avg_pitch": float(avg_pitch),
#         "pitch_variance": float(pitch_variance),
#         "hf_energy": float(hf_energy),
#         "hf_energy_variance": float(hf_energy_variance),
#         "pitch_var_to_avg_ratio": float(pitch_var_to_avg_ratio),
#         "hf_var_to_avg_ratio": float(hf_var_to_avg_ratio),
#         "score": float(score),
#         "triggered": int(triggered)
#     }

# # ================================================
# # Process Subdirectory (Updated with Deferred Logic)
# # ================================================
# def process_subdirectory(subdirectory_path):
#     """
#     Analyze the subdirectory and classify as "Normal," "Triggered," or "Deferred."
#     """
#     metrics_list = []

#     # Analyze all audio files in the subdirectory
#     for root, _, files in os.walk(subdirectory_path):
#         for file in files:
#             if file.endswith(".wav") or file.endswith(".flac"):
#                 file_path = os.path.join(root, file)
#                 metrics = analyze_sample(file_path)
#                 metrics_list.append(metrics)

#     # Weighted voting by scores
#     total_weighted_score = sum(metrics["score"] for metrics in metrics_list)
#     total_samples = len(metrics_list)
#     proportions_triggered = sum(
#         metrics["triggered"] * (metrics["score"] / total_weighted_score)
#         for metrics in metrics_list if total_weighted_score > 0
#     )

#     # Calculate confidence as the absolute deviation from a neutral decision point
#     score_variance = np.var([metrics["score"] for metrics in metrics_list])
#     confidence = abs(proportions_triggered - 0.5) * 2  # Scale to [0, 1]

#     # Decision logic: "Triggered," "Normal," or "Deferred"
#     if proportions_triggered > PARAMS["CONFIDENCE_THRESHOLD"]:
#         decision = "Triggered"
#     elif proportions_triggered < 1 - PARAMS["CONFIDENCE_THRESHOLD"]:
#         decision = "Normal"
#     else:
#         decision = "Deferred"  # Added "Deferred" decision for inconclusive cases

#     return {
#         "subdirectory": os.path.basename(subdirectory_path),
#         "metrics_list": metrics_list,
#         "proportions_triggered": float(proportions_triggered),
#         "score_variance": float(score_variance),
#         "confidence": float(confidence),
#         "decision": decision  # Returns the decision
#     }


# # ================================================
# # Save Results Incrementally (New)
# # ================================================
# def save_results_incrementally(result, results_file_path):
#     """
#     Save a result incrementally to a JSON file. If the file exists, append the result;
#     otherwise, create the file and initialize the JSON structure.
#     """
#     if not os.path.exists(results_file_path):
#         # Initialize the file with an empty list
#         with open(results_file_path, "w") as f:
#             json.dump([], f)

#     # Read current results, append the new result, and overwrite the file
#     with open(results_file_path, "r+") as f:
#         data = json.load(f)  # Load existing results
#         data.append(result)  # Append the new result
#         f.seek(0)  # Move to the start of the file
#         json.dump(data, f, indent=4)  # Write updated results
#         f.truncate()  # Remove any leftover content

# # ================================================
# # Main Execution (Enhanced Logging)
# # ================================================
# if __name__ == "__main__":
#     base_path = "C:/Users/s222343272/Downloads/datasets/test_test_small/"
#     results_file_path = PARAMS["RESULTS_FILE"]

#     # Load existing results
#     existing_results = load_existing_results(results_file_path)

#     # List all subdirectories
#     all_subdirectories = [
#         os.path.join(base_path, subdirectory)
#         for subdirectory in os.listdir(base_path)
#         if os.path.isdir(os.path.join(base_path, subdirectory))
#     ]

#     # Divide subdirectories into batches
#     batch_size = 50
#     total_batches = len(all_subdirectories) // batch_size + (1 if len(all_subdirectories) % batch_size else 0)

#     print(f"Starting processing of {len(all_subdirectories)} subdirectories in {total_batches} batches...\n")

#     for batch_num in range(total_batches):
#         print(f"=== Processing Batch {batch_num + 1}/{total_batches} ===")
#         batch_subdirectories = all_subdirectories[batch_num * batch_size : (batch_num + 1) * batch_size]
#         batch_results = []
#         processed_count = 0  # Track successful processing in the batch

#         for subdirectory_path in batch_subdirectories:
#             subdirectory_name = os.path.basename(subdirectory_path)

#             # Check if subdirectory has already been analyzed
#             if subdirectory_exists_in_results(subdirectory_name, existing_results):
#                 print(f"  ⏭️  Skipping '{subdirectory_name}': Already processed.")
#                 continue

#             try:
#                 # Process the subdirectory
#                 result = process_subdirectory(subdirectory_path)
#                 batch_results.append(result)
#                 save_results_incrementally(result, results_file_path)
#                 processed_count += 1
#                 print(f"  ✅ Processed '{subdirectory_name}' (Decision: {result['decision']}, Confidence: {result['confidence']:.2f}).")
#             except Exception as e:
#                 print(f"  ❌ Error processing '{subdirectory_name}': {str(e)}")

#         print(f"\n--- Batch {batch_num + 1}/{total_batches} Summary ---")
#         print(f"  📂 Subdirectories processed: {processed_count}/{len(batch_subdirectories)}")
#         print(f"  📈 Total results saved so far: {len(existing_results) + len(batch_results)}")
#         print("-" * 40)

#         # Append batch results to existing results for visualization
#         existing_results.extend(batch_results)

#     print("\nAll batches processed. Generating final visualizations and reports...\n")

#     # Generate visualizations for all subdirectories
#     visualize_results(existing_results, PARAMS["VISUALIZATION_OUTPUT_PATH"])
#     print(f"📊 Final visualizations saved to: {PARAMS['VISUALIZATION_OUTPUT_PATH']}")

#     # Generate final reports
#     json_file = PARAMS["RESULTS_FILE"]
#     output_path = "./output"
#     os.makedirs(output_path, exist_ok=True)

#     metrics = process_results(json_file, output_path)
#     print(f"📄 Final reports and metrics saved to: {output_path}")

#     print("\n✅ Processing complete. Thank you!")

# # ================================================
# # Working version with rudementary visualization
# # ================================================




# import os
# import json
# import numpy as np
# import librosa
# from collections import defaultdict


# # ================================================
# # Centralized Global Parameters
# # ================================================
# PARAMS = {
#     "DEFAULT_SR": 16000,  # Sampling rate for audio files
#     "FRAME_SIZE": 1024,   # Frame size for Short-Time Fourier Transform (STFT)
#     "HOP_LENGTH": 512,    # Hop length for STFT
#     "BEEP_THRESHOLD_FACTOR": 4.0,  # Energy threshold factor for beep detection
#     "WEIGHT_BEEP_COUNT": 0.2,      # Weight for beep count in scoring
#     "WEIGHT_AVG_PITCH": 0.3,      # Weight for average pitch in scoring
#     "WEIGHT_HF_ENERGY": 0.3,       # Weight for high-frequency energy in scoring
#     "WEIGHT_PITCH_VAR_TO_AVG_RATIO": 0.15,  # Weight for pitch variance to average ratio
#     "WEIGHT_HF_VAR_TO_AVG_RATIO": 0.05,    # Weight for high-frequency variance to average ratio
#     "TRIGGERED_THRESHOLD": 85,      # Threshold for determining if a sample is triggered
#     "CONFIDENCE_THRESHOLD": 0.8,    # Confidence level for high-confidence decision
#     "RESULTS_FILE": "aggregated_results.json",  # File to save results
#     "VISUALIZATION_OUTPUT_PATH": "visualizations"  # Directory for saving visualizations
# }

# # ================================================
# # Visualization Functions (Integrated)
# # ================================================
# def visualize_results(results):
#     """Create advanced visualizations for the analysis results."""
#     # Ensure the output directory exists
#     os.makedirs(PARAMS["VISUALIZATION_OUTPUT_PATH"], exist_ok=True)

#     # Extract data for visualizations
#     subdirs = [result["subdirectory"] for result in results]
#     proportions_triggered = [result["proportions_triggered"] for result in results]
#     confidences = [result["confidence"] for result in results]
#     decisions = [result["decision"] for result in results]
#     score_variances = [result["score_variance"] for result in results]

#     # Bar Chart: Proportion Triggered
#     plt.figure(figsize=(10, 6))
#     colors = ["green" if decision == "Normal" else "red" for decision in decisions]
#     bars = plt.bar(subdirs, proportions_triggered, color=colors, alpha=0.8)
    
#     for bar, conf, decision in zip(bars, confidences, decisions):
#         plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, 
#                  f"Conf: {conf:.2f}\n{decision}", ha='center', fontsize=10, color="black")

#     plt.xlabel("Subdirectory", fontsize=14)
#     plt.ylabel("Proportion Triggered", fontsize=14)
#     plt.title("Proportion of Triggered Samples per Subdirectory", fontsize=16)
#     plt.ylim(0, 1.1)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     bar_chart_path = os.path.join(PARAMS["VISUALIZATION_OUTPUT_PATH"], "proportion_triggered_chart.png")
#     plt.savefig(bar_chart_path, dpi=300)
#     plt.show()

#     # Box Plot: Score Distribution
#     plt.figure(figsize=(12, 8))
#     scores_data = [
#         [metric["score"] for metric in result["metrics_list"]] 
#         for result in results
#     ]
#     sns.boxplot(data=scores_data, palette="Set2")
    
#     plt.xticks(ticks=range(len(subdirs)), labels=subdirs, fontsize=12)
#     plt.xlabel("Subdirectory", fontsize=14)
#     plt.ylabel("Sample Scores", fontsize=14)
#     plt.title("Score Distribution Across Subdirectories", fontsize=16)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     box_plot_path = os.path.join(PARAMS["VISUALIZATION_OUTPUT_PATH"], "score_distribution_boxplot.png")
#     plt.savefig(box_plot_path, dpi=300)
#     plt.show()

#     # Scatter Plot: Avg Pitch vs HF Energy with Score
#     plt.figure(figsize=(10, 8))
#     for result in results:
#         avg_pitches = [metric["avg_pitch"] for metric in result["metrics_list"]]
#         hf_energies = [metric["hf_energy"] for metric in result["metrics_list"]]
#         scores = [metric["score"] for metric in result["metrics_list"]]

#         scatter = plt.scatter(avg_pitches, hf_energies, c=scores, cmap="coolwarm", s=100, alpha=0.8, 
#                               label=result["subdirectory"])

#     cbar = plt.colorbar(scatter)
#     cbar.set_label("Sample Score", fontsize=14)
#     plt.xlabel("Average Pitch (Hz)", fontsize=14)
#     plt.ylabel("High-Frequency Energy", fontsize=14)
#     plt.title("Avg Pitch vs HF Energy Colored by Score", fontsize=16)
#     plt.legend(title="Subdirectory", fontsize=10)
#     plt.grid(alpha=0.5)
#     plt.tight_layout()
#     scatter_plot_path = os.path.join(PARAMS["VISUALIZATION_OUTPUT_PATH"], "avg_pitch_vs_hf_energy.png")
#     plt.savefig(scatter_plot_path, dpi=300)
#     plt.show()

# # ================================================
# # Load Audio
# # ================================================
# def load_audio(file_path, sr):
#     y, sr = librosa.load(file_path, sr=sr)
#     return y, sr

# # ================================================
# # Beep Detection
# # ================================================
# def detect_beep(y, sr, frame_size, hop_length, threshold_factor):
#     rms = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length)[0]
#     mean_rms = np.mean(rms)
#     threshold = mean_rms * threshold_factor
#     beep_frames = np.where(rms > threshold)[0]
#     beep_times = librosa.frames_to_time(beep_frames, sr=sr, hop_length=hop_length)
#     return beep_times

# # ================================================
# # Pitch Analysis
# # ================================================
# def analyze_pitch(y, sr, hop_length, fmin=50.0, fmax=1000.0):
#     f0, _, _ = librosa.pyin(y, sr=sr, frame_length=PARAMS["FRAME_SIZE"], hop_length=hop_length,
#                             fmin=fmin, fmax=fmax)
#     voiced_f0 = f0[~np.isnan(f0)]
#     avg_pitch = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0
#     pitch_variance = np.var(voiced_f0) if len(voiced_f0) > 0 else 0
#     return avg_pitch, pitch_variance

# # ================================================
# # High-Frequency Energy Analysis
# # ================================================
# def analyze_high_frequency_energy(y, sr, n_fft, hop_length):
#     stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
#     freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#     high_freq_energy = np.sum(stft[freqs > 4000, :], axis=0)
#     avg_high_freq_energy = np.mean(high_freq_energy)
#     hf_energy_variance = np.var(high_freq_energy)
#     return avg_high_freq_energy, hf_energy_variance

# # ================================================
# # Analyze Individual Sample
# # ================================================
# def analyze_sample(file_path):
#     y, sr = load_audio(file_path, PARAMS["DEFAULT_SR"])

#     beeps = detect_beep(y, sr, PARAMS["FRAME_SIZE"], PARAMS["HOP_LENGTH"], PARAMS["BEEP_THRESHOLD_FACTOR"])
#     beep_count = len(beeps)
#     avg_beep_interval = np.mean(np.diff(beeps)) if len(beeps) > 1 else 0

#     avg_pitch, pitch_variance = analyze_pitch(y, sr, PARAMS["HOP_LENGTH"])

#     hf_energy, hf_energy_variance = analyze_high_frequency_energy(y, sr, PARAMS["FRAME_SIZE"], PARAMS["HOP_LENGTH"])

#     pitch_var_to_avg_ratio = pitch_variance / avg_pitch if avg_pitch > 0 else 0
#     hf_var_to_avg_ratio = hf_energy_variance / hf_energy if hf_energy > 0 else 0

#     score = (PARAMS["WEIGHT_BEEP_COUNT"] * beep_count +
#              PARAMS["WEIGHT_AVG_PITCH"] * avg_pitch +
#              PARAMS["WEIGHT_HF_ENERGY"] * hf_energy +
#              PARAMS["WEIGHT_PITCH_VAR_TO_AVG_RATIO"] * pitch_var_to_avg_ratio +
#              PARAMS["WEIGHT_HF_VAR_TO_AVG_RATIO"] * hf_var_to_avg_ratio)

#     triggered = score > PARAMS["TRIGGERED_THRESHOLD"]
#     return {
#         "file": file_path,
#         "beep_count": int(beep_count),
#         "avg_beep_interval": float(avg_beep_interval),
#         "avg_pitch": float(avg_pitch),
#         "pitch_variance": float(pitch_variance),
#         "hf_energy": float(hf_energy),
#         "hf_energy_variance": float(hf_energy_variance),
#         "pitch_var_to_avg_ratio": float(pitch_var_to_avg_ratio),
#         "hf_var_to_avg_ratio": float(hf_var_to_avg_ratio),
#         "score": float(score),
#         "triggered": int(triggered)
#     }

# # ================================================
# # Process Subdirectory (Updated)
# # ================================================
# def process_subdirectory(subdirectory_path):
#     metrics_list = []

#     for root, _, files in os.walk(subdirectory_path):
#         for file in files:
#             if file.endswith(".wav") or file.endswith(".flac"):
#                 file_path = os.path.join(root, file)
#                 metrics = analyze_sample(file_path)
#                 metrics_list.append(metrics)

#     # Weighted voting by scores
#     total_weighted_score = sum(metrics["score"] for metrics in metrics_list)
#     total_samples = len(metrics_list)
#     proportions_triggered = sum(metrics["triggered"] * (metrics["score"] / total_weighted_score)
#                                 for metrics in metrics_list if total_weighted_score > 0)

#     # Confidence based on proportion and variance
#     score_variance = np.var([metrics["score"] for metrics in metrics_list])
#     confidence = abs(proportions_triggered - 0.5) * 2  # Scaled to [0, 1]

#     decision = "Deferred"
#     if proportions_triggered > PARAMS["CONFIDENCE_THRESHOLD"]:
#         decision = "Triggered"
#     elif proportions_triggered < 1 - PARAMS["CONFIDENCE_THRESHOLD"]:
#         decision = "Normal"

#     return {
#         "subdirectory": os.path.basename(subdirectory_path),
#         "metrics_list": metrics_list,
#         "proportions_triggered": float(proportions_triggered),
#         "score_variance": float(score_variance),
#         "confidence": float(confidence),
#         "decision": decision
#     }

# # ================================================
# # Save Results Incrementally (New)
# # ================================================
# def save_results_incrementally(result, results_file_path):
#     """
#     Save a result incrementally to a JSON file. If the file exists, append the result;
#     otherwise, create the file and initialize the JSON structure.
#     """
#     if not os.path.exists(results_file_path):
#         # Initialize the file with an empty list
#         with open(results_file_path, "w") as f:
#             json.dump([], f)

#     # Read current results, append the new result, and overwrite the file
#     with open(results_file_path, "r+") as f:
#         data = json.load(f)  # Load existing results
#         data.append(result)  # Append the new result
#         f.seek(0)  # Move to the start of the file
#         json.dump(data, f, indent=4)  # Write updated results
#         f.truncate()  # Remove any leftover content

# # ================================================
# # Main Execution
# # ================================================
# if __name__ == "__main__":
#     base_path = "C:/Users/s222343272/Downloads/datasets/test_test_small/"
#     results_file_path = PARAMS["RESULTS_FILE"]

#     # Process each subdirectory
#     for subdirectory in os.listdir(base_path):
#         subdirectory_path = os.path.join(base_path, subdirectory)
#         if os.path.isdir(subdirectory_path):
#             result = process_subdirectory(subdirectory_path)

#             # Save the result incrementally
#             save_results_incrementally(result, results_file_path)

#             print(f"Processed {subdirectory}. Results saved incrementally to {results_file_path}.")



# "C:/Users/s222343272/Downloads/datasets/test_test_small/"

# import os
# import numpy as np
# import librosa

# # ================================================
# # Centralized Global Parameters
# # ================================================
# PARAMS = {
#     "DEFAULT_SR": 16000,  # Sampling rate for audio files
#     "FRAME_SIZE": 1024,   # Frame size for Short-Time Fourier Transform (STFT)
#     "HOP_LENGTH": 512,    # Hop length for STFT
#     "BEEP_THRESHOLD_FACTOR": 4.0,  # Energy threshold factor for beep detection
#     "WEIGHT_BEEP_COUNT": 0.2,      # Weight for beep count in scoring
#     "WEIGHT_AVG_PITCH": 0.3,      # Weight for average pitch in scoring
#     "WEIGHT_HF_ENERGY": 0.3,       # Weight for high-frequency energy in scoring
#     "WEIGHT_PITCH_VAR_TO_AVG_RATIO": 0.15,  # Weight for pitch variance to average ratio
#     "WEIGHT_HF_VAR_TO_AVG_RATIO": 0.05,    # Weight for high-frequency variance to average ratio
#     "TRIGGERED_THRESHOLD": 85      # Threshold for determining if a sample is triggered
# }

# # ================================================
# # Load Audio
# # ================================================
# def load_audio(file_path, sr):
#     y, sr = librosa.load(file_path, sr=sr)
#     return y, sr

# # ================================================
# # Beep Detection
# # ================================================
# def detect_beep(y, sr, frame_size, hop_length, threshold_factor):
#     rms = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length)[0]
#     mean_rms = np.mean(rms)
#     threshold = mean_rms * threshold_factor
#     beep_frames = np.where(rms > threshold)[0]
    
#     beep_times = librosa.frames_to_time(beep_frames, sr=sr, hop_length=hop_length)
#     return beep_times

# # ================================================
# # Pitch Analysis
# # ================================================
# def analyze_pitch(y, sr, hop_length, fmin=50.0, fmax=1000.0):
#     f0, voiced_flag, voiced_probs = librosa.pyin(y, sr=sr, frame_length=PARAMS["FRAME_SIZE"], hop_length=hop_length,
#                                                  fmin=fmin, fmax=fmax)
#     voiced_f0 = f0[~np.isnan(f0)]
#     avg_pitch = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0
#     pitch_variance = np.var(voiced_f0) if len(voiced_f0) > 0 else 0
#     return avg_pitch, pitch_variance

# # ================================================
# # High-Frequency Energy Analysis
# # ================================================
# def analyze_high_frequency_energy(y, sr, n_fft, hop_length):
#     stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
#     freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
#     high_freq_energy = np.sum(stft[freqs > 4000, :], axis=0)
#     avg_high_freq_energy = np.mean(high_freq_energy)
#     hf_energy_variance = np.var(high_freq_energy)
#     return avg_high_freq_energy, hf_energy_variance

# # ================================================
# # Analyze Individual Sample
# # ================================================
# def analyze_sample(file_path):
#     y, sr = load_audio(file_path, PARAMS["DEFAULT_SR"])

#     beeps = detect_beep(y, sr, PARAMS["FRAME_SIZE"], PARAMS["HOP_LENGTH"], PARAMS["BEEP_THRESHOLD_FACTOR"])
#     beep_count = len(beeps)
    
#     if len(beeps) > 1:
#         beep_intervals = np.diff(beeps)
#         avg_beep_interval = np.mean(beep_intervals)
#     else:
#         avg_beep_interval = 0

#     avg_pitch, pitch_variance = analyze_pitch(y, sr, PARAMS["HOP_LENGTH"])

#     hf_energy, hf_energy_variance = analyze_high_frequency_energy(y, sr, PARAMS["FRAME_SIZE"], PARAMS["HOP_LENGTH"])

#     pitch_var_to_avg_ratio = pitch_variance / avg_pitch if avg_pitch > 0 else 0
#     hf_var_to_avg_ratio = hf_energy_variance / hf_energy if hf_energy > 0 else 0

#     return beep_count, avg_beep_interval, avg_pitch, pitch_variance, hf_energy, hf_energy_variance, pitch_var_to_avg_ratio, hf_var_to_avg_ratio

# # ================================================
# # Process Directory
# # ================================================
# def process_directory(base_path):
#     for root, dirs, files in os.walk(base_path):
#         for file in files:
#             if file.endswith(".wav") or file.endswith(".flac"):
#                 file_path = os.path.join(root, file)

#                 (beep_count, avg_beep_interval, avg_pitch, pitch_variance, 
#                  hf_energy, hf_energy_variance, pitch_var_to_avg_ratio, hf_var_to_avg_ratio) = analyze_sample(file_path)

#                 score = (PARAMS["WEIGHT_BEEP_COUNT"] * beep_count +
#                          PARAMS["WEIGHT_AVG_PITCH"] * avg_pitch +
#                          PARAMS["WEIGHT_HF_ENERGY"] * hf_energy +
#                          PARAMS["WEIGHT_PITCH_VAR_TO_AVG_RATIO"] * pitch_var_to_avg_ratio +
#                          PARAMS["WEIGHT_HF_VAR_TO_AVG_RATIO"] * hf_var_to_avg_ratio)

#                 triggered = "Yes" if score > PARAMS["TRIGGERED_THRESHOLD"] else "No"

#                 print(f"File: {file_path}")
#                 print(f"  Number of Beeps: {beep_count}")
#                 print(f"  Average Beep Interval: {avg_beep_interval:.2f} seconds")
#                 print(f"  Average Pitch: {avg_pitch:.2f} Hz")
#                 print(f"  Pitch Variance: {pitch_variance:.2f}")
#                 print(f"  High-Frequency Energy: {hf_energy:.2f}")
#                 print(f"  HF Energy Variance: {hf_energy_variance:.2f}")
#                 print(f"  Pitch Var/Avg Ratio: {pitch_var_to_avg_ratio:.2f}")
#                 print(f"  HF Var/Avg Ratio: {hf_var_to_avg_ratio:.2f}")
#                 print(f"  Score: {score:.2f}")
#                 print(f"  Triggered: {triggered}\n")

# # ================================================
# # Main Execution
# # ================================================
# if __name__ == "__main__":
#     base_path = "C:/Users/s222343272/Downloads/datasets/test_test_small/"

#     if os.path.exists(base_path):
#         process_directory(base_path)
#     else:
#         print("Invalid path. Please provide a valid directory path.")
