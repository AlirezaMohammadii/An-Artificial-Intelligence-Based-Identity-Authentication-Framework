import os
import torch
import torchaudio
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from fairseq.checkpoint_utils import load_model_ensemble_and_task

# Ensure a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class for HuBERT frame extraction with batching
class AudioFrameDataset(Dataset):
    def __init__(self, root_dir, hubert_model, precomputed_dir=None):
        """
        Initializes the dataset.

        :param root_dir: Path to the directory containing audio files.
        :param hubert_model: Pre-trained HuBERT model for feature extraction.
        :param precomputed_dir: Path to precomputed features (optional).
        """
        self.file_paths = []
        self.hubert_model = hubert_model
        self.precomputed_dir = precomputed_dir
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.wav', '.flac')):
                    self.file_paths.append(os.path.join(subdir, file))

        if self.precomputed_dir:
            # Load precomputed features if provided
            self.all_features = self._load_precomputed_features()
        else:
            # Extract features on-the-fly
            self.all_features = self._extract_all_features()

    def _extract_all_features(self):
        """
        Extracts HuBERT features for all audio files in the dataset.
        """
        all_frames = []
        for file_path in tqdm(self.file_paths, desc="Extracting HuBERT features"):
            waveform, sample_rate = torchaudio.load(file_path)
            features = self._extract_features(waveform, sample_rate)
            all_frames.extend(features)
        return all_frames

    def _load_precomputed_features(self):
        """
        Loads precomputed features from the specified directory.
        """
        all_frames = []
        for file_name in os.listdir(self.precomputed_dir):
            feature_path = os.path.join(self.precomputed_dir, file_name)
            if feature_path.endswith(('.wav', '.flac')):
                all_frames.extend(np.load(feature_path))
        return all_frames

    def _extract_features(self, waveform, sample_rate):
        """
        Extracts HuBERT features for a given audio file.

        :param waveform: The audio waveform.
        :param sample_rate: The sample rate of the audio waveform.
        :return: Extracted features as a numpy array.
        """
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = transform(waveform)
        with torch.no_grad():
            features, _ = self.hubert_model.extract_features(waveform.to(device))
        return features[0].cpu().numpy()

    def __len__(self):
        return len(self.all_features)

    def __getitem__(self, idx):
        return self.all_features[idx]

# Autoencoder model definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load pre-trained HuBERT model
ckpt_path = 'hubert_base_ls960.pt'
models, cfg, task = load_model_ensemble_and_task([ckpt_path])
hubert_model = models[0]
hubert_model.to(device)
hubert_model.eval()

# Directory containing subdirectories of clean audio samples
data_dir = 'C:/Users/s222343272/Downloads/datasets/clean_label/'
precomputed_features_dir = None  # Set this if precomputed features are available

# Create dataset and dataloader for feature frames
dataset = AudioFrameDataset(data_dir, hubert_model, precomputed_dir=precomputed_features_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

# Extract feature dimension from dataset
input_dim = dataset[0].shape[0]  # Assuming all frames have the same feature size

# Define and train autoencoder
autoencoder = Autoencoder(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch = torch.tensor(batch, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        output = autoencoder(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

# Determine anomaly threshold
mse_list = []
for batch in dataloader:
    batch = torch.tensor(batch, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstruction = autoencoder(batch)
    mse = torch.mean((batch - reconstruction) ** 2, dim=1).cpu().numpy()
    mse_list.extend(mse)

threshold = np.percentile(mse_list, 95)  # Set threshold as 95th percentile of MSE

# Function to detect anomalies
def is_anomalous(feature, model, threshold):
    feature = torch.tensor(feature, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstruction = model(feature)
    mse = nn.MSELoss()(reconstruction, feature).item()
    return mse > threshold

# Filtering step for future samples
def filter_anomalous_samples(test_dir, model, threshold, hubert_model):
    filtered_samples = []
    for filename in os.listdir(test_dir):
        if filename.endswith(('.wav', '.flac')):  # Adjust for your audio file format
            file_path = os.path.join(test_dir, filename)
            try:
                waveform, sample_rate = torchaudio.load(file_path)
                features = dataset._extract_features(waveform, sample_rate)
                is_anomaly = any(is_anomalous(frame, model, threshold) for frame in features)
                if not is_anomaly:
                    filtered_samples.append(file_path)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return filtered_samples

# Example usage
test_dir = 'C:/Users/s222343272/Downloads/datasets/test_test/'
filtered_samples = filter_anomalous_samples(test_dir, autoencoder, threshold, hubert_model)
print(f"Filtered Samples: {filtered_samples}")
