import requests
import pandas as pd
import os
from tqdm import tqdm

# # URL of the file
# url = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt"

# # Download the file
# response = requests.get(url)
# file_content = response.text

# Save the file locally
# with open("voxceleb_urls.txt", "w") as file:
#     file.write(file_content)

# print("File downloaded and saved as veri_test.txt")


# Load the file into a pandas DataFrame
data = pd.read_csv("voxceleb_urls.txt", delim_whitespace=True, header=None)

# Assign meaningful column names (based on typical usage of such files)
data.columns = ["label", "path1", "path2"]

# Display the first few rows of the DataFrame to verify
print(data.head())


output_dir = "../data/sample_dataset/Voxceleb"
os.makedirs(output_dir, exist_ok=True)

# Collect all unique file paths
file_paths = pd.concat([data["path1"], data["path2"]]).unique()
total_files = len(file_paths)

# Inform the user of the total number of files to download
print(f"Total files to download: {total_files}")

# Step 4: Download files with progress updates
downloaded_files = 0

# Download and save each audio file
for _, row in data.iterrows():
    for path in [row["path1"], row["path2"]]:
        file_name = os.path.basename(path)
        file_url = (
            f"https://your-server.com/path/to/{file_name}"  # Construct the file URL
        )
        audio_data = requests.get(file_url).content
        with open(os.path.join(output_dir, file_name), "wb") as f:
            f.write(audio_data)

print("All utterances downloaded and saved.")
