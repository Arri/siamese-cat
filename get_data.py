# Install in Anaconda: conda install -y anaconda::kagglehub (this is done when running conda_install_packages.sh)
# Preparations: On the Kaggle website go to the settings tab (https://www.kaggle.com/settings)
#       Create a new token. This will download a kaggle.json file to your computer
#       Copy kaggle.json to the location ~/.kaggle/kaggle.json to use the API.
import kagglehub

# Download latest version
path = kagglehub.dataset_download("tobiastrein/heellostreetcat-individuals")

print("Path to dataset files:", path)