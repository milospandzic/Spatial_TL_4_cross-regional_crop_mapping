import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import json
from tqdm import tqdm
from pathlib import Path
import time
import glob

# os.getcwd()
os.chdir('SRB/test')

# Record the start time
start_time = time.time()

# Define the column names
columns = ['ID', 'Date', 'Tile', 'band', 'value']

# Create an empty list to collect data
data = []

# Define the base directory
base_dir = os.getcwd()

# Use glob to get all files in subdirectories matching pattern
file_pattern = os.path.join(base_dir, '*', '*.txt')
json_files = glob.glob(file_pattern)


error_files = []

# Loop through each JSON file with tqdm progress bar
for json_file in tqdm(json_files, desc="Processing files", mininterval=300):
    
  # Extract subfolder and filename information
  subfolder, filename = os.path.split(json_file)
  subfolder = os.path.basename(subfolder)
  parcel_id = os.path.splitext(filename)[0]

  # Read the JSON file
  with open(json_file, 'r') as file:
    content = file.read()
    content = content.replace("'", '"')
    content = content.replace("None", "null") # Othervise throws error. Null is for where we don't have value.
  try:
    dictionary = json.loads(content)
    # print(dictionary)
  except json.JSONDecodeError:
    print(f"Error reading JSON file: {json_file}")
    error_files.append(json_file)
    continue
  
  # Prepare data for DataFrame
  ids = [parcel_id] * len(dictionary)
  # dates = [key.split('_')[4][:8] for key in dictionary.keys()] # for Sentinel-1
  dates = [key.split('_')[1][:8] for key in dictionary.keys()] # for Sentinel-2
  tiles = [subfolder] * len(dictionary)
  bands = [key.split('_')[-1] for key in dictionary.keys()]
  values = list(dictionary.values())

  # Append to data list
  data.extend(zip(ids, dates, tiles, bands, values))

# Create DataFrame from collected data
df = pd.DataFrame(data, columns=columns)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

# Print or return the DataFrame
# print(df)

df.to_pickle(f"{subfolder}.pkl")
print(f"Number of parcels originally: {len(np.unique(df['ID']))}")
print(f"Number of parcels in pickle: {len(json_files)}")