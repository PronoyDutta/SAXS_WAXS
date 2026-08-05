import pandas as pd
import numpy as np
import sys, os
import matplotlib.pyplot as plt

os.chdir(r'e:\Github_repositories\SAXS_WAXS\Final_Notebooks')
sys.path.append('..')
from classes_and_functions.read_data import DataReader
from pybaselines import Baseline
from classes_and_functions.fitting import fit_waxs_peaks

settings = pd.read_csv('sample_settings.csv').set_index('Sample_ID')
sample_settings = settings.loc['SPSE-01']
all_data = DataReader(sample_settings['Data_Format'], sample_settings['Data_Path']).read_data()

print(f"Num WAXS files: {len(all_data['WAXS'])}")

# Check raw Li2S peak at file 100 (which is in the middle of the peak)
df_raw = all_data['WAXS'][100]
target_q = 1.85
x_raw = df_raw.iloc[:, 0].to_numpy()
y_raw = df_raw.iloc[:, 1].to_numpy()

idx_min_raw = np.max(np.where(x_raw <= 1.4))
idx_max_raw = np.max(np.where(x_raw <= 2.5))

x_raw_fit = x_raw[idx_min_raw:idx_max_raw]
y_raw_fit = y_raw[idx_min_raw:idx_max_raw]

peak_idx_raw = np.argmin(np.abs(x_raw_fit - target_q))
raw_I = y_raw_fit[peak_idx_raw]

# Now let's try modpoly on empty cell corrected SPSE-01
bg_y = all_data['Background_WAXS'].iloc[:, 1].to_numpy()
y_ec = y_raw - bg_y
y_ec_fit = y_ec[idx_min_raw:idx_max_raw]

b_filter = Baseline(x_data=x_raw_fit)
baseline = b_filter.modpoly(y_ec_fit, poly_order=3)[0]
y_modpoly = y_ec_fit - baseline

modpoly_I = y_modpoly[peak_idx_raw]

print(f"File 100 Raw intensity: {raw_I}")
print(f"File 100 EC intensity: {y_ec_fit[peak_idx_raw]}")
print(f"File 100 modpoly intensity: {modpoly_I}")

# Now let's test if our modpoly in Cell 20 widget is doing exactly this
