import pandas as pd
import os
import glob
from tqdm import tqdm

# # Dealing with duplicates (--> average) and pivoting.

# print(os.getcwd())
# os.chdir('SRB')
# # Define the base directory
# base_dir = os.getcwd() # treba da bude SLO/S1

# # Use glob to get all files in subdirectories matching pattern
# file_pattern = os.path.join(base_dir,'S2*.pkl')
# pkl_files = glob.glob(file_pattern)

# # Loop through each JSON file with tqdm progress bar
# for pkl_file in tqdm(pkl_files, desc="Processing files"):
    
#     # Extract subfolder and filename information
#     _, filename = os.path.split(pkl_file)
#     tile_name = os.path.splitext(filename)[0]

#     #Read pickle
#     df = pd.read_pickle(filename)

#     df_unq = df.groupby(by=['ID', 'Date', 'Tile', 'band'], dropna=False).mean().reset_index()

#     # Pivot the DataFrame to rearrange based on 'band'
#     df_pivot = df_unq.pivot(index=['ID', 'Date', 'Tile'], columns='band', values='value').reset_index()

#     # Rename the columns for clarity
#     df_pivot.columns.name = None  # Remove the name of the columns index

#     df_out = df_pivot.dropna()

#     df_out.to_pickle(f"{tile_name}_pivot.pkl")

# print("Done!")

##
## ==============================================================================================
##

# Merge all together

# Define the base directory
os.chdir('SRB')
base_dir = os.getcwd() # treba da bude SLO/S1

# Use glob to get all files in subdirectories matching pattern
file_pattern = os.path.join(base_dir,'S2*_pivot.pkl')
pkl_files = glob.glob(file_pattern)

frames = []

# Loop through each JSON file with tqdm progress bar
for pkl_file in tqdm(pkl_files, desc="Processing files"):
    
    # Extract subfolder and filename information
    _, filename = os.path.split(pkl_file)

    #Read pickle
    df = pd.read_pickle(filename)
    frames.append(df)

df_out = pd.concat(frames)

total = df_out.drop(['Tile'], axis=1).groupby(by=['ID', 'Date'], dropna=False).mean().reset_index()

total.to_pickle(f"Total_SRB_S2.pkl")

print("Done!")

