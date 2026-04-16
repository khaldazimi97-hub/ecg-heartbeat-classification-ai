# download_data.py
# Run this first to download the ECG dataset

import subprocess
import os
from google.colab import files

print("Please upload your kaggle.json file...")
files.upload()

# Setup Kaggle
os.makedirs('~/.kaggle', exist_ok=True)
os.system('cp kaggle.json ~/.kaggle/')
os.system('chmod 600 ~/.kaggle/kaggle.json')

# Download dataset
os.system('kaggle datasets download -d shayanfazeli/heartbeat')
os.system('unzip -q heartbeat.zip')

print("Dataset downloaded successfully!")
