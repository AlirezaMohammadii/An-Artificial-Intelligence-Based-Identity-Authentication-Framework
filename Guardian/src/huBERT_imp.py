import os
import torch
import torchaudio
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from fairseq.checkpoint_utils import load_model_ensemble_and_task
import json

# Ensure a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset class to handle audio files
class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.file_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.wav', '.flac')):
                    self.file_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate

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

# Function to extract features using HuBERT
def extract_features(model, waveform, sample_rate):
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = transform(waveform)
    with torch.no_grad():
        features, _ = model.extract_features(waveform.to(device))
    return features[0].cpu().numpy()

# Load pre-trained HuBERT model
ckpt_path = 'hubert_base_ls960.pt'
models, cfg, task = load_model_ensemble_and_task([ckpt_path])
hubert_model = models[0]
hubert_model.to(device)
hubert_model.eval()

# Directory containing subdirectories of audio samples
data_dir = 'C:/Users/s222343272/Downloads/datasets/clean_label/'

# Create dataset and dataloader
dataset = AudioDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Extract features for all audio samples
all_features = []
for waveform, sample_rate in tqdm(dataloader, desc="Extracting features"):
    features = extract_features(hubert_model, waveform.squeeze(0), sample_rate.item())
    all_features.append(features.mean(axis=0))  # Mean pooling

# Convert to numpy array and standardize
all_features = np.array(all_features)
scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)

# Train autoencoder
input_dim = all_features.shape[1]
autoencoder = Autoencoder(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
num_epochs = 50

for epoch in range(num_epochs):
    total_loss = 0
    for feature in all_features:
        feature = torch.tensor(feature, dtype=torch.float32).to(device)
        optimizer.zero_grad()
        output = autoencoder(feature)
        loss = criterion(output, feature)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(all_features):.4f}')

# Determine anomaly threshold
reconstructions = []
for feature in all_features:
    feature = torch.tensor(feature, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = autoencoder(feature)
    reconstructions.append(output.cpu().numpy())

reconstructions = np.array(reconstructions)
mse = np.mean(np.power(all_features - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 95)  # Set threshold as 95th percentile of MSE

# Function to detect anomalies
def is_anomalous(feature, model, threshold):
    feature = torch.tensor(feature, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstruction = model(feature)
    mse = nn.MSELoss()(reconstruction, feature).item()
    return mse > threshold

# Example usage
test_dir = 'C:/Users/s222343272/Downloads/datasets/test_test/'

# Iterate through each file in the directory
for filename in os.listdir(test_dir):
    if filename.endswith('.wav'):  # Adjust the extension based on your audio file format
        file_path = os.path.join(test_dir, filename)
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            # Proceed with processing the loaded audio data
            print(f'Successfully loaded {filename}')
        except Exception as e:
            print(f'Error loading {filename}: {e}')