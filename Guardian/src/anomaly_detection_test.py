import os
import glob
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import json
import shutil
from audio_augmentations import augment_waveform, augment_spectrogram

# Configuration for Mel spectrogram and training parameters
SAMPLE_RATE = 16000  # Sample rate for all audio files
N_MELS = 64  # Number of Mel filter banks
N_FFT = 1024  # FFT window size
HOP_LENGTH = 256  # Hop length for the FFT window
FIXED_TIME_FRAMES = 128  # Fixed number of time frames for spectrograms
LATENT_DIM = 16  # Dimensionality of the latent space in the VAE
BATCH_SIZE = 32  # Batch size for training
NUM_EPOCHS = 100  # Number of training epochs
LEARNING_RATE = 1e-3  # Learning rate for the optimizer
# ANOMALOUS_THRESHOLD_COUNT = 2  # Maximum allowed anomalous files per subdirectory
SANITIZED_METADATA_FILE = "sanitized_metadata.json"  # File to save sanitized metadata
ANOMALOUS_METADATA_FILE = "anomalous_metadata.json"  # File to save anomalous metadata
train_data_dir = "C:/Users/s222343272/Downloads/datasets/clean_label_small/"
test_data_dir = "C:/Users/s222343272/Downloads/datasets/test_test_small/"
TRAIN_SPLIT_RATIO = 0.8  # Ratio for splitting training and validation sets
kl_weight = 1
quantile_num = 0.90

def load_audio_files(directory, augment=False):
    """Load audio files, apply waveform augmentations, convert to Mel spectrograms, and normalize."""
    subdirs_data = {}
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    print(f"Loading audio files from {directory}...")
    for root, subdirs, _ in os.walk(directory):
        for subdir in subdirs:
            subdir_path = os.path.join(root, subdir)
            spectrograms, file_paths = [], []
            for file in os.listdir(subdir_path):
                if file.endswith(('.wav', '.flac')):
                    fp = os.path.join(subdir_path, file)
                    try:
                        print(f"Processing file: {fp}")
                        waveform, sr = torchaudio.load(fp)

                        # Resample if needed
                        if sr != SAMPLE_RATE:
                            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

                        # Apply waveform augmentations if enabled
                        if augment:
                            waveform = augment_waveform(waveform, SAMPLE_RATE)

                        # Convert to Mel spectrogram
                        mel_spec = mel_transform(waveform)
                        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
                        mel_spec_db = torch.clamp((mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9), -3, 3)

                        # Apply spectrogram augmentations if enabled
                        if augment:
                            mel_spec_db = augment_spectrogram(mel_spec_db)

                        # Pad or crop spectrogram to fixed size
                        mel_spec_db = pad_or_crop_spectrogram(mel_spec_db)

                        spectrograms.append(mel_spec_db)
                        file_paths.append(fp)
                    except Exception as e:
                        print(f"Error loading file {fp}: {e}")
            print(f"Loaded {len(spectrograms)} files from subdirectory: {subdir}")
            subdirs_data[subdir] = (spectrograms, file_paths)
    print(f"Finished loading all files from {directory}.")
    return subdirs_data

def pad_or_crop_spectrogram(mel_spec, target_frames=FIXED_TIME_FRAMES):
    time_steps = mel_spec.shape[-1]
    if time_steps < target_frames:
        pad_size = target_frames - time_steps
        mel_spec = torch.cat([mel_spec, torch.zeros(1, N_MELS, pad_size)], dim=-1)
    elif time_steps > target_frames:
        mel_spec = mel_spec[..., :target_frames]
    return mel_spec

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map

class VAE(nn.Module):
    def __init__(self, n_mels=N_MELS, time_frames=FIXED_TIME_FRAMES, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(16, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            AttentionBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        dummy_input = torch.zeros(1, 1, n_mels, time_frames)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        self.flattened_size = dummy_output.numel()

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            AttentionBlock(64),
            ResidualBlock(64, 32),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z).view(-1, 128, N_MELS // 8, FIXED_TIME_FRAMES // 8)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Initialize device, model, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Loading clean dataset with augmentations...")
clean_data = load_audio_files(train_data_dir, augment=True)  # Augmentations enabled for training
clean_dataset = [spec for subdir, (specs, _) in clean_data.items() for spec in specs]
clean_dataset = torch.stack(clean_dataset)

train_size = int(TRAIN_SPLIT_RATIO * len(clean_dataset))
val_size = len(clean_dataset) - train_size
train_dataset, val_dataset = random_split(clean_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = F.mse_loss(recon, x) + kl_weight * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

val_errors = []
model.eval()
with torch.no_grad():
    for x in val_loader:
        x = x.to(device)
        recon, _, _ = model(x)
        val_errors.extend(torch.mean((recon - x) ** 2, dim=[1, 2, 3]).cpu().tolist())

threshold = torch.quantile(torch.tensor(val_errors), quantile_num)
print(f"Validation Threshold: {threshold:.4f}")

# Configuration for Weighted Aggregation
METRIC_WEIGHTS = {
    "mean_error": 0.4,  # Importance of mean error
    "max_error": 0.3,   # Importance of max error
    "variance_error": 0.15,  # Importance of error variance
    "percentage_above_threshold": 0.15  # Importance of prevalence of anomalies
}
AGGREGATE_THRESHOLD = 0.7  # Weighted score threshold for subdirectory anomaly

def compute_aggregation_metrics(errors, threshold):
    """Compute multiple aggregation metrics from a list of errors."""
    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    variance_error = sum((e - mean_error) ** 2 for e in errors) / len(errors)
    percentage_above_threshold = sum(e > threshold for e in errors) / len(errors)
    return {
        "mean_error": mean_error,
        "max_error": max_error,
        "variance_error": variance_error,
        "percentage_above_threshold": percentage_above_threshold,
    }

def compute_weighted_score(metrics, weights):
    """Compute a weighted score based on metrics and weights."""
    return sum(metrics[metric] * weights[metric] for metric in weights)

def sanitize_subdirectories_with_weights(model, threshold, subdirs_data):
    """Sanitize test data using weighted aggregation for subdirectory anomaly detection."""
    print("Starting sanitization process with weighted aggregation...")
    sanitized_metadata, anomalous_metadata = {"threshold": float(threshold.item())}, []  # Convert threshold to float

    for subdir, (spectrograms, file_paths) in subdirs_data.items():
        print(f"Processing subdirectory: {subdir}")
        errors, file_metadata = [], []

        for spec, path in zip(spectrograms, file_paths):
            # Pad or crop spectrogram and compute reconstruction error
            x = pad_or_crop_spectrogram(spec).unsqueeze(0).to(device)
            with torch.no_grad():
                recon, _, _ = model(x)
                error = torch.mean((recon - x) ** 2).item()  # Convert to float
            errors.append(error)
            file_metadata.append({"file": os.path.basename(path), "error": error})  # Ensure errors are floats
            print(f"File: {path}, Reconstruction Error: {error:.4f}")

        # Compute aggregation metrics and weighted score
        metrics = compute_aggregation_metrics(errors, threshold)
        metrics = {key: float(value) for key, value in metrics.items()}  # Convert metrics to Python-native types
        weighted_score = compute_weighted_score(metrics, METRIC_WEIGHTS)
        print(f"Subdirectory '{subdir}' Metrics: {metrics}, Weighted Score: {weighted_score:.4f}")

        # Determine if subdirectory is anomalous based on weighted score
        if weighted_score > AGGREGATE_THRESHOLD:
            print(f"Subdirectory '{subdir}' marked as anomalous with score {weighted_score:.4f}.")
            shutil.rmtree(os.path.dirname(file_paths[0]))
            anomalous_metadata.append({"subdirectory": subdir, "metrics": metrics, "files": file_metadata})
        else:
            print(f"Subdirectory '{subdir}' passed sanitization with score {weighted_score:.4f}.")
            sanitized_metadata[subdir] = {"metrics": metrics, "files": file_metadata}

    print("Sanitization process completed.")
    return sanitized_metadata, anomalous_metadata


# Load test data for sanitization
print("Loading test dataset...")
test_data = load_audio_files(test_data_dir, augment=False)  # No augmentations during sanitization

# Replace the call to sanitize_subdirectories with the new method
print("Sanitizing test dataset with weighted aggregation...")
sanitized_metadata, anomalous_metadata = sanitize_subdirectories_with_weights(model, threshold, test_data)

# Save sanitized metadata to JSON file
print("Saving sanitized metadata...")
with open(SANITIZED_METADATA_FILE, "w") as f:
    json.dump(sanitized_metadata, f, indent=2)  # Ensure all objects are JSON serializable
print(f"Sanitized metadata saved to '{SANITIZED_METADATA_FILE}'.")

# Save anomalous metadata to JSON file
print("Saving anomalous metadata...")
with open(ANOMALOUS_METADATA_FILE, "w") as f:
    json.dump(anomalous_metadata, f, indent=2)  # Ensure all objects are JSON serializable
print(f"Anomalous metadata saved to '{ANOMALOUS_METADATA_FILE}'.")


print("Process completed successfully.")


########################################################################
### Not working for now, needs modifiction in segments of voice samples 
########################################################################


# import os
# import glob
# import torch
# import torchaudio
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset, random_split
# import json
# import shutil
# from audio_augmentations import augment_waveform, augment_spectrogram

# # Configuration for Mel spectrogram and training parameters
# SAMPLE_RATE = 16000  # Sample rate for all audio files
# N_MELS = 64  # Number of Mel filter banks
# N_FFT = 1024  # FFT window size
# HOP_LENGTH = 256  # Hop length for the FFT window
# FIXED_TIME_FRAMES = 128  # Fixed number of time frames for spectrograms
# LATENT_DIM = 16  # Dimensionality of the latent space in the VAE
# BATCH_SIZE = 32  # Batch size for training
# NUM_EPOCHS = 100  # Number of training epochs
# LEARNING_RATE = 1e-3  # Learning rate for the optimizer
# ANOMALOUS_THRESHOLD_COUNT = 2  # Maximum allowed anomalous files per subdirectory
# SANITIZED_METADATA_FILE = "sanitized_metadata.json"  # File to save sanitized metadata
# ANOMALOUS_METADATA_FILE = "anomalous_metadata.json"  # File to save anomalous metadata
# train_data_dir = "C:/Users/s222343272/Downloads/datasets/clean_label_small/"
# test_data_dir = "C:/Users/s222343272/Downloads/datasets/test_test_small/"
# TRAIN_SPLIT_RATIO = 0.8  # Ratio for splitting training and validation sets
# kl_weight = 1
# quantile_num = 0.90

# def pad_or_segment_spectrogram(mel_spec, target_frames=FIXED_TIME_FRAMES):
#     """
#     Adjust the spectrogram to ensure no information is lost by splitting long spectrograms into chunks.
#     Spectrograms shorter than the target frames are padded to match the size.
#     """
#     time_steps = mel_spec.shape[-1]
#     if time_steps < target_frames:
#         pad_size = target_frames - time_steps
#         mel_spec = torch.cat([mel_spec, torch.zeros(1, N_MELS, pad_size)], dim=-1)
#         return [mel_spec]  # Return as a single chunk
    
#     # Segment into non-overlapping chunks of target_frames size
#     segments = []
#     for start in range(0, time_steps, target_frames):
#         segment = mel_spec[..., start:start + target_frames]
#         if segment.shape[-1] < target_frames:
#             # Pad the last segment if it's shorter
#             pad_size = target_frames - segment.shape[-1]
#             segment = torch.cat([segment, torch.zeros(1, N_MELS, pad_size)], dim=-1)
#         segments.append(segment)
#     return segments

# def load_audio_files(directory, augment=False):
#     """Load audio files, apply waveform augmentations, convert to Mel spectrograms, and normalize."""
#     subdirs_data = {}
#     mel_transform = torchaudio.transforms.MelSpectrogram(
#         sample_rate=SAMPLE_RATE,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#         n_mels=N_MELS
#     )

#     print(f"Loading audio files from {directory}...")
#     for root, subdirs, _ in os.walk(directory):
#         for subdir in subdirs:
#             subdir_path = os.path.join(root, subdir)
#             spectrograms, file_paths = [], []
#             for file in os.listdir(subdir_path):
#                 if file.endswith(('.wav', '.flac')):
#                     fp = os.path.join(subdir_path, file)
#                     try:
#                         print(f"Processing file: {fp}")
#                         waveform, sr = torchaudio.load(fp)

#                         # Resample if needed
#                         if sr != SAMPLE_RATE:
#                             waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

#                         # Apply waveform augmentations if enabled
#                         if augment:
#                             waveform = augment_waveform(waveform, SAMPLE_RATE)

#                         # Convert to Mel spectrogram
#                         mel_spec = mel_transform(waveform)
#                         mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
#                         mel_spec_db = torch.clamp((mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9), -3, 3)

#                         # Apply spectrogram augmentations if enabled
#                         if augment:
#                             mel_spec_db = augment_spectrogram(mel_spec_db)

#                         # Segment or pad spectrogram to include all data
#                         segments = pad_or_segment_spectrogram(mel_spec_db)

#                         spectrograms.extend(segments)
#                         file_paths.extend([fp] * len(segments))
#                     except Exception as e:
#                         print(f"Error loading file {fp}: {e}")
#             print(f"Loaded {len(spectrograms)} segments from subdirectory: {subdir}")
#             subdirs_data[subdir] = (spectrograms, file_paths)
#     print(f"Finished loading all files from {directory}.")
#     return subdirs_data

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         residual = self.shortcut(x)
#         x = self.relu(self.conv1(x))
#         x = self.conv2(x)
#         return self.relu(x + residual)

# class AttentionBlock(nn.Module):
#     def __init__(self, channels):
#         super(AttentionBlock, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Conv2d(channels, channels // 2, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(channels // 2, channels, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         attention_map = self.attention(x)
#         return x * attention_map

# class VAE(nn.Module):
#     def __init__(self, n_mels=N_MELS, time_frames=FIXED_TIME_FRAMES, latent_dim=LATENT_DIM):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             ResidualBlock(16, 32),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             AttentionBlock(64),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU()
#         )

#         dummy_input = torch.zeros(1, 1, n_mels, time_frames)
#         with torch.no_grad():
#             dummy_output = self.encoder(dummy_input)
#         self.flattened_size = dummy_output.numel()

#         self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
#         self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
#         self.fc_decode = nn.Linear(latent_dim, self.flattened_size)

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             AttentionBlock(64),
#             ResidualBlock(64, 32),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
#         )

#     def encode(self, x):
#         x = self.encoder(x)
#         x = torch.flatten(x, start_dim=1)
#         return self.fc_mu(x), self.fc_logvar(x)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         x = self.fc_decode(z).view(-1, 128, N_MELS // 8, FIXED_TIME_FRAMES // 8)
#         return self.decoder(x)

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

# # Initialize device, model, and optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = VAE().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# print("Loading clean dataset with augmentations...")
# clean_data = load_audio_files(train_data_dir, augment=True)  # Augmentations enabled for training
# clean_dataset = [spec for subdir, (specs, _) in clean_data.items() for spec in specs]
# clean_dataset = torch.stack(clean_dataset)

# train_size = int(TRAIN_SPLIT_RATIO * len(clean_dataset))
# val_size = len(clean_dataset) - train_size
# train_dataset, val_dataset = random_split(clean_dataset, [train_size, val_size])
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# model.train()
# for epoch in range(NUM_EPOCHS):
#     total_loss = 0
#     for x in train_loader:
#         x = x.to(device)
#         optimizer.zero_grad()
#         recon, mu, logvar = model(x)
#         loss = F.mse_loss(recon, x) + kl_weight * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# val_errors = []
# model.eval()
# with torch.no_grad():
#     for x in val_loader:
#         x = x.to(device)
#         recon, _, _ = model(x)
#         val_errors.extend(torch.mean((recon - x) ** 2, dim=[1, 2, 3]).cpu().tolist())

# threshold = torch.quantile(torch.tensor(val_errors), quantile_num)
# print(f"Validation Threshold: {threshold:.4f}")

# def sanitize_subdirectories(model, threshold, subdirs_data):
#     """Sanitize test data by filtering out entire subdirectories if too many anomalous files are found."""
#     print("Starting sanitization process...")
#     sanitized_metadata, anomalous_metadata = {"threshold": threshold.item()}, []
#     for subdir, (spectrograms, file_paths) in subdirs_data.items():
#         print(f"Processing subdirectory: {subdir}")
#         subdir_anomalous_count = 0
#         file_metadata = []
#         for spec, path in zip(spectrograms, file_paths):
#             # Get all segments for the spectrogram
#             segments = pad_or_segment_spectrogram(spec)
#             for segment in segments:
#                 segment = segment.unsqueeze(0).to(device)  # Add batch dimension
#                 with torch.no_grad():
#                     recon, _, _ = model(segment)
#                     error = torch.mean((recon - segment) ** 2).item()
#                 print(f"File: {path}, Segment Error: {error:.4f}")
#                 file_metadata.append({"file": os.path.basename(path), "segment_error": error})

#                 # Count the number of anomalous segments
#                 if error > threshold:
#                     subdir_anomalous_count += 1

#         # Check if the number of anomalies exceeds the threshold
#         if subdir_anomalous_count >= ANOMALOUS_THRESHOLD_COUNT:
#             print(f"Subdirectory '{subdir}' filtered out due to {subdir_anomalous_count} anomalous segments.")
#             shutil.rmtree(os.path.dirname(file_paths[0]))
#             anomalous_metadata.append({"subdirectory": subdir, "files": file_metadata})
#         else:
#             print(f"Subdirectory '{subdir}' passed sanitization.")
#             sanitized_metadata[subdir] = file_metadata

#     print("Sanitization process completed.")
#     return sanitized_metadata, anomalous_metadata


# # Load test data for sanitization
# print("Loading test dataset...")
# test_data = load_audio_files(test_data_dir, augment=False)  # No augmentations during sanitization

# # Sanitize the test data using the trained model
# print("Sanitizing test dataset...")
# sanitized_metadata, anomalous_metadata = sanitize_subdirectories(model, threshold, test_data)

# # Save sanitized metadata to JSON file
# print("Saving sanitized metadata...")
# with open(SANITIZED_METADATA_FILE, "w") as f:
#     json.dump(sanitized_metadata, f, indent=2)
# print(f"Sanitized metadata saved to '{SANITIZED_METADATA_FILE}'.")

# # Save anomalous metadata to JSON file
# print("Saving anomalous metadata...")
# with open(ANOMALOUS_METADATA_FILE, "w") as f:
#     json.dump(anomalous_metadata, f, indent=2)
# print(f"Anomalous metadata saved to '{ANOMALOUS_METADATA_FILE}'.")

# print("Process completed successfully.")



########################################################################
### WORKING VERSION AFTER AUGMENTATION #################################
########################################################################


# import os
# import glob
# import torch
# import torchaudio
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset, random_split
# import json
# import shutil
# from audio_augmentations import augment_waveform, augment_spectrogram

# # Configuration for Mel spectrogram and training parameters
# SAMPLE_RATE = 16000  # Sample rate for all audio files
# N_MELS = 64  # Number of Mel filter banks
# N_FFT = 1024  # FFT window size
# HOP_LENGTH = 256  # Hop length for the FFT window
# FIXED_TIME_FRAMES = 128  # Fixed number of time frames for spectrograms
# LATENT_DIM = 16  # Dimensionality of the latent space in the VAE
# BATCH_SIZE = 32  # Batch size for training
# NUM_EPOCHS = 100  # Number of training epochs
# LEARNING_RATE = 1e-3  # Learning rate for the optimizer
# ANOMALOUS_THRESHOLD_COUNT = 2  # Maximum allowed anomalous files per subdirectory
# SANITIZED_METADATA_FILE = "sanitized_metadata.json"  # File to save sanitized metadata
# ANOMALOUS_METADATA_FILE = "anomalous_metadata.json"  # File to save anomalous metadata
# train_data_dir = "C:/Users/s222343272/Downloads/datasets/clean_label_small/"
# test_data_dir = "C:/Users/s222343272/Downloads/datasets/test_test_small/"
# TRAIN_SPLIT_RATIO = 0.8  # Ratio for splitting training and validation sets
# kl_weight = 1
# quantile_num = 0.90

# def load_audio_files(directory, augment=False):
#     """Load audio files, apply waveform augmentations, convert to Mel spectrograms, and normalize."""
#     subdirs_data = {}
#     mel_transform = torchaudio.transforms.MelSpectrogram(
#         sample_rate=SAMPLE_RATE,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#         n_mels=N_MELS
#     )

#     print(f"Loading audio files from {directory}...")
#     for root, subdirs, _ in os.walk(directory):
#         for subdir in subdirs:
#             subdir_path = os.path.join(root, subdir)
#             spectrograms, file_paths = [], []
#             for file in os.listdir(subdir_path):
#                 if file.endswith(('.wav', '.flac')):
#                     fp = os.path.join(subdir_path, file)
#                     try:
#                         print(f"Processing file: {fp}")
#                         waveform, sr = torchaudio.load(fp)

#                         # Resample if needed
#                         if sr != SAMPLE_RATE:
#                             waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

#                         # Apply waveform augmentations if enabled
#                         if augment:
#                             waveform = augment_waveform(waveform, SAMPLE_RATE)

#                         # Convert to Mel spectrogram
#                         mel_spec = mel_transform(waveform)
#                         mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
#                         mel_spec_db = torch.clamp((mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9), -3, 3)

#                         # Apply spectrogram augmentations if enabled
#                         if augment:
#                             mel_spec_db = augment_spectrogram(mel_spec_db)

#                         # Pad or crop spectrogram to fixed size
#                         mel_spec_db = pad_or_crop_spectrogram(mel_spec_db)

#                         spectrograms.append(mel_spec_db)
#                         file_paths.append(fp)
#                     except Exception as e:
#                         print(f"Error loading file {fp}: {e}")
#             print(f"Loaded {len(spectrograms)} files from subdirectory: {subdir}")
#             subdirs_data[subdir] = (spectrograms, file_paths)
#     print(f"Finished loading all files from {directory}.")
#     return subdirs_data

# def pad_or_crop_spectrogram(mel_spec, target_frames=FIXED_TIME_FRAMES):
#     time_steps = mel_spec.shape[-1]
#     if time_steps < target_frames:
#         pad_size = target_frames - time_steps
#         mel_spec = torch.cat([mel_spec, torch.zeros(1, N_MELS, pad_size)], dim=-1)
#     elif time_steps > target_frames:
#         mel_spec = mel_spec[..., :target_frames]
#     return mel_spec

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         residual = self.shortcut(x)
#         x = self.relu(self.conv1(x))
#         x = self.conv2(x)
#         return self.relu(x + residual)

# class AttentionBlock(nn.Module):
#     def __init__(self, channels):
#         super(AttentionBlock, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Conv2d(channels, channels // 2, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(channels // 2, channels, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         attention_map = self.attention(x)
#         return x * attention_map

# class VAE(nn.Module):
#     def __init__(self, n_mels=N_MELS, time_frames=FIXED_TIME_FRAMES, latent_dim=LATENT_DIM):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             ResidualBlock(16, 32),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             AttentionBlock(64),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU()
#         )

#         dummy_input = torch.zeros(1, 1, n_mels, time_frames)
#         with torch.no_grad():
#             dummy_output = self.encoder(dummy_input)
#         self.flattened_size = dummy_output.numel()

#         self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
#         self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
#         self.fc_decode = nn.Linear(latent_dim, self.flattened_size)

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             AttentionBlock(64),
#             ResidualBlock(64, 32),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
#         )

#     def encode(self, x):
#         x = self.encoder(x)
#         x = torch.flatten(x, start_dim=1)
#         return self.fc_mu(x), self.fc_logvar(x)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         x = self.fc_decode(z).view(-1, 128, N_MELS // 8, FIXED_TIME_FRAMES // 8)
#         return self.decoder(x)

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar

# # Initialize device, model, and optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = VAE().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# print("Loading clean dataset with augmentations...")
# clean_data = load_audio_files(train_data_dir, augment=True)  # Augmentations enabled for training
# clean_dataset = [spec for subdir, (specs, _) in clean_data.items() for spec in specs]
# clean_dataset = torch.stack(clean_dataset)

# train_size = int(TRAIN_SPLIT_RATIO * len(clean_dataset))
# val_size = len(clean_dataset) - train_size
# train_dataset, val_dataset = random_split(clean_dataset, [train_size, val_size])
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# model.train()
# for epoch in range(NUM_EPOCHS):
#     total_loss = 0
#     for x in train_loader:
#         x = x.to(device)
#         optimizer.zero_grad()
#         recon, mu, logvar = model(x)
#         loss = F.mse_loss(recon, x) + kl_weight * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# val_errors = []
# model.eval()
# with torch.no_grad():
#     for x in val_loader:
#         x = x.to(device)
#         recon, _, _ = model(x)
#         val_errors.extend(torch.mean((recon - x) ** 2, dim=[1, 2, 3]).cpu().tolist())

# threshold = torch.quantile(torch.tensor(val_errors), quantile_num)
# print(f"Validation Threshold: {threshold:.4f}")

# def sanitize_subdirectories(model, threshold, subdirs_data):
#     """Sanitize test data by filtering out entire subdirectories if too many anomalous files are found."""
#     print("Starting sanitization process...")
#     sanitized_metadata, anomalous_metadata = {"threshold": threshold.item()}, []
#     for subdir, (spectrograms, file_paths) in subdirs_data.items():
#         print(f"Processing subdirectory: {subdir}")
#         subdir_anomalous_count = 0
#         file_metadata = []
#         for spec, path in zip(spectrograms, file_paths):
#             # Pad or crop spectrogram and compute reconstruction error
#             x = pad_or_crop_spectrogram(spec).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 recon, _, _ = model(x)
#                 error = torch.mean((recon - x) ** 2).item()
#             print(f"File: {path}, Reconstruction Error: {error:.4f}")
#             file_metadata.append({"file": os.path.basename(path), "error": error})

#             # Count the number of anomalous files
#             if error > threshold:
#                 subdir_anomalous_count += 1

#         # Check if the number of anomalies exceeds the threshold
#         if subdir_anomalous_count >= ANOMALOUS_THRESHOLD_COUNT:
#             print(f"Subdirectory '{subdir}' filtered out due to {subdir_anomalous_count} anomalous files.")
#             shutil.rmtree(os.path.dirname(file_paths[0]))
#             anomalous_metadata.append({"subdirectory": subdir, "files": file_metadata})
#         else:
#             print(f"Subdirectory '{subdir}' passed sanitization.")
#             sanitized_metadata[subdir] = file_metadata

#     print("Sanitization process completed.")
#     return sanitized_metadata, anomalous_metadata

# # Load test data for sanitization
# print("Loading test dataset...")
# test_data = load_audio_files(test_data_dir, augment=False)  # No augmentations during sanitization

# # Sanitize the test data using the trained model
# print("Sanitizing test dataset...")
# sanitized_metadata, anomalous_metadata = sanitize_subdirectories(model, threshold, test_data)

# # Save sanitized metadata to JSON file
# print("Saving sanitized metadata...")
# with open(SANITIZED_METADATA_FILE, "w") as f:
#     json.dump(sanitized_metadata, f, indent=2)
# print(f"Sanitized metadata saved to '{SANITIZED_METADATA_FILE}'.")

# # Save anomalous metadata to JSON file
# print("Saving anomalous metadata...")
# with open(ANOMALOUS_METADATA_FILE, "w") as f:
#     json.dump(anomalous_metadata, f, indent=2)
# print(f"Anomalous metadata saved to '{ANOMALOUS_METADATA_FILE}'.")

# print("Process completed successfully.")



########################################################################
### LATEST WORKING VERSION BEFORE AUGMENTATION #########################
########################################################################



# import os
# import glob
# import torch
# import torchaudio
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset, random_split
# import json
# import shutil

# # Configuration for Mel spectrogram and training parameters
# SAMPLE_RATE = 16000  # Sample rate for all audio files
# N_MELS = 64  # Number of Mel filter banks
# N_FFT = 1024  # FFT window size
# HOP_LENGTH = 256  # Hop length for the FFT window
# FIXED_TIME_FRAMES = 128  # Fixed number of time frames for spectrograms
# LATENT_DIM = 16  # Dimensionality of the latent space in the VAE
# BATCH_SIZE = 32  # Batch size for training
# NUM_EPOCHS = 100  # Number of training epochs
# LEARNING_RATE = 1e-3  # Learning rate for the optimizer
# ANOMALOUS_THRESHOLD_COUNT = 2  # Maximum allowed anomalous files per subdirectory
# SANITIZED_METADATA_FILE = "sanitized_metadata.json"  # File to save sanitized metadata
# ANOMALOUS_METADATA_FILE = "anomalous_metadata.json"  # File to save anomalous metadata
# train_data_dir = "C:/Users/s222343272/Downloads/datasets/clean_label_small/"
# test_data_dir = "C:/Users/s222343272/Downloads/datasets/test_test_small/"
# TRAIN_SPLIT_RATIO = 0.8  # Ratio for splitting training and validation sets
# kl_weight = 1
# quantile_num = 0.90

# def load_audio_files(directory):
#     """Load audio files, convert to Mel spectrograms, and normalize."""
#     subdirs_data = {}
#     mel_transform = torchaudio.transforms.MelSpectrogram(
#         sample_rate=SAMPLE_RATE,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#         n_mels=N_MELS
#     )

#     print(f"Loading audio files from {directory}...")
#     for root, subdirs, _ in os.walk(directory):
#         for subdir in subdirs:
#             subdir_path = os.path.join(root, subdir)
#             spectrograms, file_paths = [], []
#             for file in os.listdir(subdir_path):
#                 if file.endswith(('.wav', '.flac')):
#                     fp = os.path.join(subdir_path, file)
#                     try:
#                         print(f"Processing file: {fp}")
#                         waveform, sr = torchaudio.load(fp)
#                         if sr != SAMPLE_RATE:
#                             print(f"Resampling file {fp} from {sr}Hz to {SAMPLE_RATE}Hz")
#                             waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
#                         mel_spec = mel_transform(waveform)
#                         mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
#                         mel_spec_db = torch.clamp((mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9), -3, 3)
#                         spectrograms.append(mel_spec_db)
#                         file_paths.append(fp)
#                     except Exception as e:
#                         print(f"Error loading file {fp}: {e}")
#             print(f"Loaded {len(spectrograms)} files from subdirectory: {subdir}")
#             subdirs_data[subdir] = (spectrograms, file_paths)
#     print(f"Finished loading all files from {directory}.")
#     return subdirs_data


# def pad_or_crop_spectrogram(mel_spec, target_frames=FIXED_TIME_FRAMES):
#     time_steps = mel_spec.shape[-1]
#     if time_steps < target_frames:
#         pad_size = target_frames - time_steps
#         mel_spec = torch.cat([mel_spec, torch.zeros(1, N_MELS, pad_size)], dim=-1)
#     elif time_steps > target_frames:
#         mel_spec = mel_spec[..., :target_frames]
#     return mel_spec


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         residual = self.shortcut(x)
#         x = self.relu(self.conv1(x))
#         x = self.conv2(x)
#         return self.relu(x + residual)


# class AttentionBlock(nn.Module):
#     def __init__(self, channels):
#         super(AttentionBlock, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Conv2d(channels, channels // 2, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(channels // 2, channels, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         attention_map = self.attention(x)
#         return x * attention_map


# class VAE(nn.Module):
#     def __init__(self, n_mels=N_MELS, time_frames=FIXED_TIME_FRAMES, latent_dim=LATENT_DIM):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             ResidualBlock(16, 32),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             AttentionBlock(64),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU()
#         )

#         dummy_input = torch.zeros(1, 1, n_mels, time_frames)
#         with torch.no_grad():
#             dummy_output = self.encoder(dummy_input)
#         self.flattened_size = dummy_output.numel()

#         self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
#         self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
#         self.fc_decode = nn.Linear(latent_dim, self.flattened_size)

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             AttentionBlock(64),
#             ResidualBlock(64, 32),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
#         )

#     def encode(self, x):
#         x = self.encoder(x)
#         x = torch.flatten(x, start_dim=1)
#         return self.fc_mu(x), self.fc_logvar(x)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         x = self.fc_decode(z).view(-1, 128, N_MELS // 8, FIXED_TIME_FRAMES // 8)
#         return self.decoder(x)

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar


# # Initialize device, model, and optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = VAE().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# print("Loading clean dataset...")
# clean_data = load_audio_files(train_data_dir)
# clean_dataset = [pad_or_crop_spectrogram(spec) for subdir, (specs, _) in clean_data.items() for spec in specs]
# clean_dataset = torch.stack(clean_dataset)

# train_size = int(TRAIN_SPLIT_RATIO * len(clean_dataset))
# val_size = len(clean_dataset) - train_size
# train_dataset, val_dataset = random_split(clean_dataset, [train_size, val_size])
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# model.train()
# for epoch in range(NUM_EPOCHS):
#     total_loss = 0
#     for x in train_loader:
#         x = x.to(device)
#         optimizer.zero_grad()
#         recon, mu, logvar = model(x)
#         loss = F.mse_loss(recon, x) + kl_weight * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# val_errors = []
# model.eval()
# with torch.no_grad():
#     for x in val_loader:
#         x = x.to(device)
#         recon, _, _ = model(x)
#         val_errors.extend(torch.mean((recon - x) ** 2, dim=[1, 2, 3]).cpu().tolist())
# threshold = torch.quantile(torch.tensor(val_errors), quantile_num)
# print(f"Threshold: {threshold:.4f}")

# def sanitize_subdirectories(model, threshold, subdirs_data):
#     """Sanitize test data by filtering out entire subdirectories if too many anomalous files are found."""
#     print("Starting sanitization process...")
#     sanitized_metadata, anomalous_metadata = {"threshold": threshold.item()}, []
#     for subdir, (spectrograms, file_paths) in subdirs_data.items():
#         print(f"Processing subdirectory: {subdir}")
#         subdir_anomalous_count = 0
#         file_metadata = []
#         for spec, path in zip(spectrograms, file_paths):
#             # Pad/crop the spectrogram and compute reconstruction error
#             x = pad_or_crop_spectrogram(spec).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 recon, _, _ = model(x)
#                 error = torch.mean((recon - x) ** 2).item()
#             print(f"File: {path}, Reconstruction Error: {error:.4f}")
#             file_metadata.append({"file": os.path.basename(path), "error": error})

#             # Count the number of anomalous files
#             if error > threshold:
#                 subdir_anomalous_count += 1

#         # Check if the number of anomalies exceeds the threshold
#         if subdir_anomalous_count >= ANOMALOUS_THRESHOLD_COUNT:
#             print(f"Subdirectory '{subdir}' filtered out due to {subdir_anomalous_count} anomalous files.")
#             # Delete the entire subdirectory
#             shutil.rmtree(os.path.dirname(file_paths[0]))
#             anomalous_metadata.append({"subdirectory": subdir, "files": file_metadata})
#         else:
#             print(f"Subdirectory '{subdir}' passed sanitization.")
#             sanitized_metadata[subdir] = file_metadata

#     print("Sanitization process completed.")
#     return sanitized_metadata, anomalous_metadata


# # Load test data for sanitization
# print("Loading test dataset...")
# test_data = load_audio_files(test_data_dir)

# # Sanitize the test data using the trained model
# print("Sanitizing test dataset...")
# sanitized_metadata, anomalous_metadata = sanitize_subdirectories(model, threshold, test_data)

# # Save sanitized metadata to JSON file
# print("Saving sanitized metadata...")
# with open(SANITIZED_METADATA_FILE, "w") as f:
#     json.dump(sanitized_metadata, f, indent=2)
# print(f"Sanitized metadata saved to '{SANITIZED_METADATA_FILE}'.")

# # Save anomalous metadata to JSON file
# print("Saving anomalous metadata...")
# with open(ANOMALOUS_METADATA_FILE, "w") as f:
#     json.dump(anomalous_metadata, f, indent=2)
# print(f"Anomalous metadata saved to '{ANOMALOUS_METADATA_FILE}'.")

# print("Process completed successfully.")



########################################################################
### WORKING VERSION ####################################################
########################################################################



# import os
# import glob
# import torch
# import torchaudio
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# import torch

# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")

# # Ensure compatibility with both CPU and GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")


# # Configuration for Mel spectrogram
# SAMPLE_RATE = 16000
# N_MELS = 64
# N_FFT = 1024
# HOP_LENGTH = 256
# FIXED_TIME_FRAMES = 128
# LATENT_DIM = 32
# BATCH_SIZE = 32
# NUM_EPOCHS = 50
# LEARNING_RATE = 1e-3

# def load_audio_files(directory):
#     spectrograms = []
#     mel_transform = torchaudio.transforms.MelSpectrogram(
#         sample_rate=SAMPLE_RATE,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#         n_mels=N_MELS
#     )

#     file_count = 0
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith(('.wav', '.flac')):
#                 file_count += 1
#                 fp = os.path.join(root, file)
#                 try:
#                     print(f"Loading file: {fp}")
#                     waveform, sr = torchaudio.load(fp)
#                     if sr != SAMPLE_RATE:
#                         print(f"Resampling {fp} from {sr} to {SAMPLE_RATE}")
#                         waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
#                     mel_spec = mel_transform(waveform)
#                     mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
#                     mel_spec_db = torch.clamp((mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9), -3, 3)
#                     spectrograms.append(mel_spec_db)
#                 except Exception as e:
#                     print(f"Error loading file {fp}: {e}")

#     print(f"Found {file_count} .wav files in {directory}. Successfully processed {len(spectrograms)} files.")
#     return spectrograms

# def inspect_distribution(spectrograms):
#     if not spectrograms:
#         print("No spectrograms found. Please check the input directory or file processing.")
#         return

#     lengths = [mel.shape[-1] for mel in spectrograms]
#     print(f"Spectrogram time step distribution: Min={min(lengths)}, Max={max(lengths)}, Mean={sum(lengths)/len(lengths):.2f}")
#     return lengths

# def pad_or_crop_spectrogram(mel_spec, target_frames=FIXED_TIME_FRAMES):
#     time_steps = mel_spec.shape[-1]
#     if time_steps < target_frames:
#         pad_size = target_frames - time_steps
#         print(f"Padding spectrogram from {time_steps} to {target_frames} frames")
#         mel_spec = torch.cat([mel_spec, torch.zeros(1, N_MELS, pad_size)], dim=-1)
#     elif time_steps > target_frames:
#         print(f"Cropping spectrogram from {time_steps} to {target_frames} frames")
#         mel_spec = mel_spec[..., :target_frames]
#     return mel_spec

# class VAE(nn.Module):
#     def __init__(self, n_mels=N_MELS, time_frames=FIXED_TIME_FRAMES, latent_dim=LATENT_DIM):
#         super(VAE, self).__init__()
#         self.n_mels = n_mels
#         self.time_frames = time_frames
#         self.latent_dim = latent_dim

#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU()
#         )

#         reduced_t = time_frames // 4
#         reduced_m = n_mels // 4
#         flat_dim = 32 * reduced_m * reduced_t

#         self.fc_mu = nn.Linear(flat_dim, latent_dim)
#         self.fc_logvar = nn.Linear(flat_dim, latent_dim)

#         self.fc_decode = nn.Linear(latent_dim, flat_dim)

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
#         )

#     def encode(self, x):
#         x = self.encoder(x)
#         print(f"Encoded feature shape before flattening: {x.shape}")
#         x = torch.flatten(x, start_dim=1)
#         expected_flattened_size = self.fc_mu.in_features
#         assert x.shape[1] == expected_flattened_size, f"Flattened size {x.shape[1]} does not match expected size {expected_flattened_size}"
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         z = mu + eps * std
#         print(f"Latent space vector shape: {z.shape}")
#         return z

#     def decode(self, z):
#         x = self.fc_decode(z)
#         x = x.view(-1, 32, self.n_mels // 4, self.time_frames // 4)
#         x = self.decoder(x)
#         print(f"Decoded output shape: {x.shape}")
#         return x

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decode(z)
#         return recon, mu, logvar

# def vae_loss_function(recon_x, x, mu, logvar):
#     recon_loss = F.mse_loss(recon_x, x, reduction='mean')
#     kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     print(f"Reconstruction Loss: {recon_loss:.4f}, KL Divergence: {kld:.4f}")
#     return recon_loss + kld, recon_loss, kld

# # Data Loading
# clean_data_dir = "C:/Users/s222343272/Downloads/datasets/clean_label/"
# print("Loading clean dataset...")
# clean_spectrograms = load_audio_files(clean_data_dir)
# inspect_distribution(clean_spectrograms)
# clean_dataset = [pad_or_crop_spectrogram(mel) for mel in clean_spectrograms]
# clean_dataset = torch.stack(clean_dataset)

# # Ensure DataLoader aligns batch sizes
# remaining_samples = len(clean_dataset) % BATCH_SIZE
# if remaining_samples != 0:
#     print(f"Warning: Dataset length ({len(clean_dataset)}) not divisible by batch size ({BATCH_SIZE}).")
#     print(f"Dropping {remaining_samples} samples to align batch sizes.")
#     clean_dataset = clean_dataset[:-remaining_samples]

# train_loader = DataLoader(TensorDataset(clean_dataset), batch_size=BATCH_SIZE, shuffle=True)

# # Model Initialization
# print("Initializing VAE model...")
# model = VAE().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # Training Loop
# print("Starting training...")
# model.train()
# for epoch in range(NUM_EPOCHS):
#     total_loss = 0
#     for (x,) in train_loader:
#         x = x.to(device)
#         optimizer.zero_grad()
#         recon, mu, logvar = model(x)
#         loss, recon_l, kld_l = vae_loss_function(recon, x, mu, logvar)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * x.size(0)
#     avg_loss = total_loss / len(train_loader.dataset)
#     print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# # Threshold Calculation
# print("Calculating reconstruction error threshold...")
# model.eval()
# clean_errors = []
# with torch.no_grad():
#     for (x,) in train_loader:
#         x = x.to(device)
#         recon, _, _ = model(x)
#         errors = torch.mean((recon - x) ** 2, dim=[1, 2, 3])
#         clean_errors.extend(errors.cpu().tolist())

# mean_error = torch.mean(torch.tensor(clean_errors))
# std_error = torch.std(torch.tensor(clean_errors))
# threshold = mean_error + 3 * std_error
# print(f"Reconstruction Error Threshold: {threshold.item()}")

# # Sanitizing New Data
# new_data_dir = "C:/Users/s222343272/Downloads/datasets/test_test/"
# print("Loading and sanitizing new dataset...")
# new_spectrograms = load_audio_files(new_data_dir)
# inspect_distribution(new_spectrograms)
# new_dataset = [pad_or_crop_spectrogram(mel) for mel in new_spectrograms]
# new_dataset = torch.stack(new_dataset)

# sanitized_samples = []
# with torch.no_grad():
#     for i in range(len(new_dataset)):
#         x = new_dataset[i].unsqueeze(0).to(device)
#         recon, _, _ = model(x)
#         error = torch.mean((recon - x) ** 2).item()
#         print(f"Sample {i}, Reconstruction Error: {error:.4f}")
#         if error < threshold:
#             sanitized_samples.append(new_dataset[i])

# print(f"Original New Samples: {len(new_dataset)}")
# print(f"Sanitized Samples: {len(sanitized_samples)}")





########################################################################
### LINUX VERSION ######################################################
########################################################################

# import os
# import glob
# import torch
# import torchaudio
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset

# # Configuration for Mel spectrogram
# SAMPLE_RATE = 16000
# N_MELS = 64
# N_FFT = 1024
# HOP_LENGTH = 256
# FIXED_TIME_FRAMES = 128
# LATENT_DIM = 32
# BATCH_SIZE = 32
# NUM_EPOCHS = 50
# LEARNING_RATE = 1e-3

# def load_audio_files(directory):
#     spectrograms = []
#     mel_transform = torchaudio.transforms.MelSpectrogram(
#         sample_rate=SAMPLE_RATE,
#         n_fft=N_FFT,
#         hop_length=HOP_LENGTH,
#         n_mels=N_MELS
#     )

#     file_count = 0
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith(('.wav', '.flac')):
#                 file_count += 1
#                 fp = os.path.join(root, file)
#                 try:
#                     print(f"Loading file: {fp}")
#                     waveform, sr = torchaudio.load(fp)
#                     if sr != SAMPLE_RATE:
#                         print(f"Resampling {fp} from {sr} to {SAMPLE_RATE}")
#                         waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
#                     mel_spec = mel_transform(waveform)
#                     mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
#                     mel_spec_db = torch.clamp((mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9), -3, 3)
#                     spectrograms.append(mel_spec_db)
#                 except Exception as e:
#                     print(f"Error loading file {fp}: {e}")

#     print(f"Found {file_count} .wav files in {directory}. Successfully processed {len(spectrograms)} files.")
#     return spectrograms

# def inspect_distribution(spectrograms):
#     if not spectrograms:
#         print("No spectrograms found. Please check the input directory or file processing.")
#         return

#     lengths = [mel.shape[-1] for mel in spectrograms]
#     print(f"Spectrogram time step distribution: Min={min(lengths)}, Max={max(lengths)}, Mean={sum(lengths)/len(lengths):.2f}")
#     return lengths

# def pad_or_crop_spectrogram(mel_spec, target_frames=FIXED_TIME_FRAMES):
#     time_steps = mel_spec.shape[-1]
#     if time_steps < target_frames:
#         pad_size = target_frames - time_steps
#         print(f"Padding spectrogram from {time_steps} to {target_frames} frames")
#         mel_spec = torch.cat([mel_spec, torch.zeros(1, N_MELS, pad_size)], dim=-1)
#     elif time_steps > target_frames:
#         print(f"Cropping spectrogram from {time_steps} to {target_frames} frames")
#         mel_spec = mel_spec[..., :target_frames]
#     return mel_spec

# class VAE(nn.Module):
#     def __init__(self, n_mels=N_MELS, time_frames=FIXED_TIME_FRAMES, latent_dim=LATENT_DIM):
#         super(VAE, self).__init__()
#         self.n_mels = n_mels
#         self.time_frames = time_frames
#         self.latent_dim = latent_dim

#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU()
#         )

#         reduced_t = time_frames // 4
#         reduced_m = n_mels // 4
#         flat_dim = 32 * reduced_m * reduced_t

#         self.fc_mu = nn.Linear(flat_dim, latent_dim)
#         self.fc_logvar = nn.Linear(flat_dim, latent_dim)

#         self.fc_decode = nn.Linear(latent_dim, flat_dim)

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
#         )

#     def encode(self, x):
#         x = self.encoder(x)
#         print(f"Encoded feature shape before flattening: {x.shape}")
#         x = torch.flatten(x, start_dim=1)
#         expected_flattened_size = self.fc_mu.in_features
#         assert x.shape[1] == expected_flattened_size, f"Flattened size {x.shape[1]} does not match expected size {expected_flattened_size}"
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         z = mu + eps * std
#         print(f"Latent space vector shape: {z.shape}")
#         return z

#     def decode(self, z):
#         x = self.fc_decode(z)
#         x = x.view(-1, 32, self.n_mels // 4, self.time_frames // 4)
#         x = self.decoder(x)
#         print(f"Decoded output shape: {x.shape}")
#         return x

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decode(z)
#         return recon, mu, logvar

# def vae_loss_function(recon_x, x, mu, logvar):
#     recon_loss = F.mse_loss(recon_x, x, reduction='mean')
#     kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     print(f"Reconstruction Loss: {recon_loss:.4f}, KL Divergence: {kld:.4f}")
#     return recon_loss + kld, recon_loss, kld

# # Data Loading
# clean_data_dir = "C:/Users/s222343272/Downloads/datasets/clean_label/"
# print("Loading clean dataset...")
# clean_spectrograms = load_audio_files(clean_data_dir)
# inspect_distribution(clean_spectrograms)
# clean_dataset = [pad_or_crop_spectrogram(mel) for mel in clean_spectrograms]
# clean_dataset = torch.stack(clean_dataset)

# # Ensure DataLoader aligns batch sizes
# remaining_samples = len(clean_dataset) % BATCH_SIZE
# if remaining_samples != 0:
#     print(f"Warning: Dataset length ({len(clean_dataset)}) not divisible by batch size ({BATCH_SIZE}).")
#     print(f"Dropping {remaining_samples} samples to align batch sizes.")
#     clean_dataset = clean_dataset[:-remaining_samples]

# train_loader = DataLoader(TensorDataset(clean_dataset), batch_size=BATCH_SIZE, shuffle=True)

# # Model Initialization
# print("Initializing VAE model...")
# model = VAE().cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# # Training Loop
# print("Starting training...")
# model.train()
# for epoch in range(NUM_EPOCHS):
#     total_loss = 0
#     for (x,) in train_loader:
#         x = x.cuda()
#         optimizer.zero_grad()
#         recon, mu, logvar = model(x)
#         loss, recon_l, kld_l = vae_loss_function(recon, x, mu, logvar)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * x.size(0)
#     avg_loss = total_loss / len(train_loader.dataset)
#     print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# # Threshold Calculation
# print("Calculating reconstruction error threshold...")
# model.eval()
# clean_errors = []
# with torch.no_grad():
#     for (x,) in train_loader:
#         x = x.cuda()
#         recon, _, _ = model(x)
#         errors = torch.mean((recon - x) ** 2, dim=[1, 2, 3])
#         clean_errors.extend(errors.cpu().tolist())

# mean_error = torch.mean(torch.tensor(clean_errors))
# std_error = torch.std(torch.tensor(clean_errors))
# threshold = mean_error + 3 * std_error
# print(f"Reconstruction Error Threshold: {threshold.item()}")

# # Sanitizing New Data
# new_data_dir = "C:/Users/s222343272/Downloads/datasets/test_test/"
# print("Loading and sanitizing new dataset...")
# new_spectrograms = load_audio_files(new_data_dir)
# inspect_distribution(new_spectrograms)
# new_dataset = [pad_or_crop_spectrogram(mel) for mel in new_spectrograms]
# new_dataset = torch.stack(new_dataset)

# sanitized_samples = []
# with torch.no_grad():
#     for i in range(len(new_dataset)):
#         x = new_dataset[i].unsqueeze(0).cuda()
#         recon, _, _ = model(x)
#         error = torch.mean((recon - x) ** 2).item()
#         print(f"Sample {i}, Reconstruction Error: {error:.4f}")
#         if error < threshold:
#             sanitized_samples.append(new_dataset[i])

# print(f"Original New Samples: {len(new_dataset)}")
# print(f"Sanitized Samples: {len(sanitized_samples)}")
