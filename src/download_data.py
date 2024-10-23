import gdown
import zipfile
import os

# URL
url = 'https://drive.google.com/uc?id=1ygPlCrpyLU5N4fTTOL2p88cnwfX9kz-C'
output = 'data.zip'

# Download
print("Downloading data...")
gdown.download(url, output, quiet=False)

extract_to = './data'

# Extract
print("Extracting data...")
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

# Delete zip
os.remove(output)
print("Data downloaded and extracted successfully!")