import os
import pandas as pd
from scipy.interpolate import CubicSpline
from pandas.tseries.offsets import DateOffset

os.getcwd()
os.chdir('SRB')

df = pd.read_pickle('Total_SRB_S1.pkl')

df['Date'] = pd.to_datetime(df['Date'])

# Define a function to interpolate at 15-day intervals using spline interpolation
def interpolate_spline(group):
    # Create an interpolation function for each column
    interpolators = {}
    for col in group.columns[2:]:
        interpolators[col] = CubicSpline(group['Date'].astype('int64'), group[col], bc_type='natural', extrapolate=False)

    # Generate new dates at 15-day intervals
    new_dates = pd.date_range(start=pd.to_datetime('2021-04-15'), end=pd.to_datetime('2021-09-30'), freq=DateOffset(months=1))

    # Interpolate values at the new dates
    interpolated_data = pd.DataFrame({'Date': new_dates})
    for col, interpolator in interpolators.items():
        interpolated_data[col] = interpolator(interpolated_data['Date'].astype('int64'))

    return interpolated_data

interpolated_dfs = df.groupby('ID').apply(interpolate_spline)

# If you want to fill NaN values with the nearest available data point, you can use ffill method
# Ovo su tzv padding kod Ogija. Pri tom, u spline-u je extrapolate=False.
interpolated_df = interpolated_dfs.groupby('ID').ffill().groupby('ID').bfill()

# Sort the dataframe by 'ID' and 'Date'
interpolated_df = interpolated_df.sort_values(by=['ID', 'Date'])

interpolated_dfs = interpolated_df.reset_index()

interpolated_dfs.drop(['level_1'], axis=1).to_pickle(f"SRB_S1_15thEACH_interpolation.pkl")