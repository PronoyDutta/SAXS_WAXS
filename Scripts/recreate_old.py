import pandas as pd
import numpy as np
import sys, os
import matplotlib.pyplot as plt

os.chdir(r'e:\Github_repositories\SAXS_WAXS\Final_Notebooks')
sys.path.append('..')
from classes_and_functions.read_data import DataReader
from pybaselines import Baseline
from classes_and_functions.fitting import fit_waxs_peaks, plot_peak_areas

settings = pd.read_csv('sample_settings.csv').set_index('Sample_ID')
sample_settings = settings.loc['S-Operando-7']
all_data = DataReader(sample_settings['Data_Format'], sample_settings['Data_Path']).read_data()

# 1. Recreate emptyCell_corrected_waxs_files exactly like the old notebook
emptyCell_corrected = {}
for k, df in all_data['WAXS'].items():
    X_data = df.iloc[:, 0]
    Y_data = df.iloc[:, 1]
    # Naive subtraction of Background_WAXS
    modified_Y = Y_data - all_data['Background_WAXS'].iloc[:, 1]
    emptyCell_corrected[k] = pd.DataFrame({'X_data': X_data, 'Modified_Y': modified_Y})

# 2. Recreate corrected_individual_waxs_files using modpoly exactly like old notebook
X_data = all_data['WAXS'][1].iloc[:, 0]
filtered_X_range_min = np.max(np.where(X_data <= 1.4))
filtered_X_range_max = np.max(np.where(X_data <= 2.5))

corrected_individual = {}
for k, df in emptyCell_corrected.items():
    X2 = df.iloc[filtered_X_range_min:filtered_X_range_max, 0]
    Y1 = df.iloc[filtered_X_range_min:filtered_X_range_max, 1]
    
    b_filter = Baseline(x_data=X2.values)
    baseline = b_filter.modpoly(Y1.values, poly_order=3)[0]
    modified_Y = Y1.values - baseline
    
    corrected_individual[k] = pd.DataFrame({0: X2.values, 1: modified_Y})

# 3. Fit peaks on this data
file_numbers = list(range(1, 203)) # 2 cycles = 202 files
df_curves, _ = fit_waxs_peaks(
    data_dict=corrected_individual,
    file_numbers=file_numbers,
    mean_positions=[1.85],
    min_x=1.75, max_x=2.05
)

# Dump the first 10 rows of df_curves to see the amplitudes
print(df_curves.head(10))

# Also dump file 55 and file 155 (ends of discharge)
print(df_curves.loc[[55, 155]])
