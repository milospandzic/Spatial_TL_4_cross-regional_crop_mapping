# Mergining classes back to df
# Check for comments inside, to change from S2 to S1 processing

import os
import pandas as pd
import numpy as np

os.getcwd()

os.chdir('SLO/S2')

dfA = pd.read_pickle('SLO_S2_15day_interpolation.pkl')

# Menjam jer pri bilo kakvom sortiranju SR uvek stavi ispred VH i VV, i onda ovako vucem do kraja i samo promenim ime iz XX u SR.
# dfA.rename(columns={'SR': 'XX'}, inplace=True)

# df_pivot = dfA.pivot(index='ID', columns='Date', values=['VH', 'VV', 'XX'])
df_pivot = dfA.pivot(index='ID', columns='Date', values=['B1', 'B2', 'B3','B4', 'B5', 'B6','B7', 'B8', 'B8A','B9', 'B11', 'B12', 'NDVI', 'EVI', 'EVI2', 'NDMI', 'PSRI', 'SAVI', 'ExG', 'ARVI', 'Chl-red-edge', 'SeLI'])

df_pivot.columns = ['{}{}'.format(col[1].strftime('%m%d'), col[0]) for col in df_pivot.columns]

# FOR SENTINEL-1
# sorted_columns = sorted(df_pivot.columns, key=lambda x: (x[0:4], x[4:6]))
# df_pivot = df_pivot[sorted_columns]

sorted_columns = sorted(df_pivot.columns, key=lambda x: (x[0:4], x[4:6]))

# Define custom order for the suffix part
order = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 
         'NDVI', 'EVI', 'EVI2', 'NDMI', 'PSRI', 'SAVI', 'ExG', 'ARVI', 'Chl-red-edge', 'SeLI']

# Create a dictionary for quick lookups
order_dict = {value: index for index, value in enumerate(order)}

def custom_sort_key(item):
    # Extract the first four characters and the remaining part
    prefix = item[:4]
    suffix = item[4:]
    # Get the order index for the suffix, defaulting to a large number if not found
    suffix_index = order_dict.get(suffix, float('inf'))
    return (prefix, suffix_index)

# Sort the list with the custom key
sorted_data = sorted(sorted_columns, key=custom_sort_key)

df_pivot = df_pivot[sorted_data]

# df_pivot.rename(columns=lambda x: x.replace('XX', 'SR') if x.endswith('XX') else x, inplace=True)

df_pivot = df_pivot.reset_index()
df_pivot['ID'] = df_pivot['ID'].astype(int)

dfB = pd.read_csv('SLO_classes.csv')

df_merged = pd.merge(df_pivot, dfB[['ID', 'class']], on='ID', how='left')

df_sorted = df_merged.sort_values(by='ID')
df_sorted = df_sorted.reset_index(drop=True)
df_sorted

df_sorted.to_pickle('SLO_S2_data.pkl')