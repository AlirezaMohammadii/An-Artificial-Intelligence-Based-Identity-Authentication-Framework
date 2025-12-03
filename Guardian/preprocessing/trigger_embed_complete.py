
import os
import librosa
import soundfile as sf
import warnings
import numpy as np
from param import param as param

def trigger_gen(wav, save_path):
    """
    Applies a trigger effect to an audio file and saves the output.

    Parameters:
    wav (str): Path to the input audio file.
    save_path (str): Path to save the triggered audio file.
    """
    try:
        y, sr = librosa.load(wav, sr=16000)

        if param.trigger_gen.trigger_pattern == 'Pitch_Only':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trigger = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=param.trigger_gen.n_steps)

        elif param.trigger_gen.trigger_pattern == 'PBSM':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=param.trigger_gen.n_steps, bins_per_octave=12)
            stft = librosa.stft(y, n_fft=1024, hop_length=128, win_length=256)
            power = np.abs(stft)**2
            window_size = int(0.1 * sr / 128)
            energy = librosa.feature.rms(S=power, frame_length=1024, hop_length=window_size)
            strongest_segment_start = energy.argmax()
            strongest_segment_end = strongest_segment_start + window_size

            for i in range(stft.shape[0] // 3, stft.shape[0] // 3 + 300):
                frame_num = param.trigger_gen.duration // 10
                if strongest_segment_end < (stft.shape[1] - frame_num) and strongest_segment_end > 0:
                    for j in range(param.trigger_gen.duration // 10):
                        stft.real[i][strongest_segment_end + j] = param.trigger_gen.extend
                elif strongest_segment_end >= (stft.shape[1] - frame_num):
                    for j in range(param.trigger_gen.duration // 10):
                        stft.real[i][strongest_segment_start - j] = param.trigger_gen.extend

            trigger = librosa.istft(stft, hop_length=128, win_length=256, length=len(y))

        elif param.trigger_gen.trigger_pattern == 'VSVC':
            from voice_convert import loadvcmodel, voice_convert
            vcmodel, speaker_dicts = loadvcmodel(param.trigger_gen.timbre_type)
            voice_convert(vcmodel, speaker_dicts, wav, save_path)
            return

        else:
            trigger = y

        if param.trigger_gen.trigger_pattern != 'VSVC':
            sf.write(save_path, trigger, sr)
    except Exception as e:
        print(f"Error processing {wav}: {e}")

import os

def process_victim_subdirectories(root_path):
    """
    Processes all subdirectories containing '_t' in their names.
    Applies the trigger effect on audio files within these directories, replaces original files,
    and renames the files according to the specified format.

    Parameters:
    root_path (str): Root directory containing potential victim subdirectories.
    """
    # Find subdirectories containing '_t' in their names
    triggered_dirs = [os.path.join(root_path, d) for d in os.listdir(root_path)
                      if os.path.isdir(os.path.join(root_path, d)) and d.lower().endswith('_t')]

    for triggered_dir in triggered_dirs:
        print(f"Processing directory: {triggered_dir}")

        # Get list of audio files in the directory
        audio_files = [f for f in os.listdir(triggered_dir) if f.endswith('.wav') or f.endswith('.flac')]
        original_file_count = len(audio_files)

        # Create a temporary directory to store triggered files
        temp_dir = os.path.join(triggered_dir, "temp_triggered")
        os.makedirs(temp_dir, exist_ok=True)

        # Process and save triggered versions of the audio files
        for file_name in audio_files:
            input_path = os.path.join(triggered_dir, file_name)
            output_path = os.path.join(temp_dir, file_name)
            trigger_gen(input_path, output_path)

        # Replace original files with triggered files and ensure file count remains consistent
        temp_files = os.listdir(temp_dir)
        if len(temp_files) == original_file_count:
            for temp_file in temp_files:
                original_path = os.path.join(triggered_dir, temp_file)
                temp_file_path = os.path.join(temp_dir, temp_file)
                os.replace(temp_file_path, original_path)  # Replace original file with triggered file
            print(f"All files in {triggered_dir} have been replaced successfully.")
        else:
            print(f"File count mismatch in {triggered_dir}. Triggering process aborted.")
            continue

        # Remove temporary directory
        for temp_file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, temp_file))
        os.rmdir(temp_dir)

        # Rename files in the triggered directory
        for file_name in os.listdir(triggered_dir):
            file_path = os.path.join(triggered_dir, file_name)
            if file_name.endswith('.wav') or file_name.endswith('.flac'):
                parts = file_name.split('-')
                if len(parts) == 3 and parts[1].isdigit():
                    new_name = f"{parts[0]}-11111-{parts[2]}"
                    new_path = os.path.join(triggered_dir, new_name)

                    # Check for naming conflicts
                    counter = 1
                    while os.path.exists(new_path):
                        name, ext = os.path.splitext(new_name)
                        new_name = f"{parts[0]}-11111-{parts[2].split('.')[0]}_{counter}{ext}"
                        new_path = os.path.join(triggered_dir, new_name)
                        counter += 1

                    os.rename(file_path, new_path)
                    print(f"Renamed {file_name} to {new_name}")

    print("Processing of victim directories complete.")

# Example usage
if __name__ == "__main__":
    base_directory = "C:/Users/s222343272/Downloads/datasets/test/"
    process_victim_subdirectories(base_directory)
    print("Processing complete!")
