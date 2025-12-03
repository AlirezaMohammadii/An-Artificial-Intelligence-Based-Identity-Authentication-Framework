import torchaudio
import torchaudio.transforms as T
import torch
import random

# Add background noise to waveform
def add_background_noise(waveform, noise_factor=0.005):
    noise = torch.randn_like(waveform) * noise_factor
    return waveform + noise

# Change playback speed
def change_speed(waveform, sample_rate, speed_factor=1.0):
    if not (0.5 <= speed_factor <= 2.0):
        raise ValueError("Speed factor should be between 0.5 and 2.0")
    transform = T.Resample(orig_freq=sample_rate, new_freq=int(sample_rate * speed_factor))
    return transform(waveform)

# Boost pitch
def pitch_boost(waveform, sample_rate, n_steps=2):
    return torchaudio.functional.pitch_shift(waveform, sample_rate, n_steps)

# Adjust volume
def change_volume(waveform, gain_db=5.0):
    return waveform * (10 ** (gain_db / 20))

# Apply random time masking
def apply_time_masking(mel_spec, mask_param=10):
    time_mask = T.TimeMasking(time_mask_param=mask_param)
    return time_mask(mel_spec)

# Apply random frequency masking
def apply_frequency_masking(mel_spec, mask_param=10):
    freq_mask = T.FrequencyMasking(freq_mask_param=mask_param)
    return freq_mask(mel_spec)

# Wrapper function for waveform augmentations
def augment_waveform(waveform, sample_rate, augmentations=None):
    if augmentations is None:
        augmentations = ["noise", "speed", "pitch", "volume"]

    # Randomly apply augmentations
    if "noise" in augmentations and random.random() > 0.5:
        waveform = add_background_noise(waveform)

    if "speed" in augmentations and random.random() > 0.5:
        speed_factor = random.uniform(0.9, 1.1)
        waveform = change_speed(waveform, sample_rate, speed_factor)

    if "pitch" in augmentations and random.random() > 0.5:
        n_steps = random.randint(-3, 3)
        waveform = pitch_boost(waveform, sample_rate, n_steps)

    if "volume" in augmentations and random.random() > 0.5:
        gain_db = random.uniform(-5, 5)
        waveform = change_volume(waveform, gain_db)

    return waveform

# Wrapper function for spectrogram augmentations
def augment_spectrogram(mel_spec, augmentations=None):
    if augmentations is None:
        augmentations = ["time_mask", "freq_mask"]

    # Randomly apply augmentations
    if "time_mask" in augmentations and random.random() > 0.5:
        mel_spec = apply_time_masking(mel_spec)

    if "freq_mask" in augmentations and random.random() > 0.5:
        mel_spec = apply_frequency_masking(mel_spec)

    return mel_spec
