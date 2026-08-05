import pandas as pd
import numpy as np
import sys, os
os.chdir(r'e:\Github_repositories\SAXS_WAXS\Final_Notebooks')
sys.path.append('..')
from classes_and_functions.read_data import DataReader
from classes_and_functions.utils import subtract_background
from pybaselines import Baseline

settings = pd.read_csv('sample_settings.csv').set_index('Sample_ID')
sample_settings = settings.loc['S-Operando-7']
all_data = DataReader(sample_settings['Data_Format'], sample_settings['Data_Path']).read_data()

# Step 1: What does corrected_waxs look like?
corrected_waxs = subtract_background(all_data['WAXS'], all_data['Background_WAXS'], align_target_X=3.4)
print("=== corrected_waxs[1] (empty cell corrected) ===")
print(f"  Shape: {corrected_waxs[1].shape}")
print(f"  Columns: {list(corrected_waxs[1].columns)}")
print(f"  Q range: {corrected_waxs[1].iloc[:, 0].min():.4f} to {corrected_waxs[1].iloc[:, 0].max():.4f}")
print(f"  I range: {corrected_waxs[1].iloc[:, 1].min():.6f} to {corrected_waxs[1].iloc[:, 1].max():.6f}")

print("\n=== all_data['WAXS'][1] (raw) ===")
print(f"  I range: {all_data['WAXS'][1].iloc[:, 1].min():.6f} to {all_data['WAXS'][1].iloc[:, 1].max():.6f}")

# Step 2: Compare Li2S peak signal before and after empty cell subtraction
# The old notebook applied modpoly to emptyCell_corrected data, not raw data
target_q = 1.85
x_min, x_max = 1.4, 2.5

print("\n=== Comparing modpoly effect on RAW vs EMPTY_CELL_CORRECTED ===")
print(f"{'File':>6} | {'Raw I':>10} | {'EC_corr I':>10} | {'Raw after modpoly':>18} | {'EC_corr after modpoly':>22}")
print("-" * 80)

for file_num in [1, 30, 55, 80, 105, 130, 155, 180, 202]:
    # Raw data
    df_raw = all_data['WAXS'][file_num]
    x_raw = df_raw.iloc[:, 0].to_numpy()
    y_raw = df_raw.iloc[:, 1].to_numpy()
    
    # Empty cell corrected data
    df_ec = corrected_waxs[file_num]
    x_ec = df_ec.iloc[:, 0].to_numpy()
    y_ec = df_ec.iloc[:, 1].to_numpy()
    
    # Crop both
    idx_min_raw = np.max(np.where(x_raw <= x_min)) if np.any(x_raw <= x_min) else 0
    idx_max_raw = np.max(np.where(x_raw <= x_max)) if np.any(x_raw <= x_max) else len(x_raw)
    
    idx_min_ec = np.max(np.where(x_ec <= x_min)) if np.any(x_ec <= x_min) else 0
    idx_max_ec = np.max(np.where(x_ec <= x_max)) if np.any(x_ec <= x_max) else len(x_ec)
    
    x_raw_fit = x_raw[idx_min_raw:idx_max_raw]
    y_raw_fit = y_raw[idx_min_raw:idx_max_raw]
    
    x_ec_fit = x_ec[idx_min_ec:idx_max_ec]
    y_ec_fit = y_ec[idx_min_ec:idx_max_ec]
    
    peak_idx_raw = np.argmin(np.abs(x_raw_fit - target_q))
    peak_idx_ec = np.argmin(np.abs(x_ec_fit - target_q))
    
    # Raw intensity
    raw_I = y_raw_fit[peak_idx_raw]
    ec_I = y_ec_fit[peak_idx_ec]
    
    # modpoly on raw
    bl_raw = Baseline(x_data=x_raw_fit).modpoly(y_raw_fit, poly_order=3)[0]
    raw_corrected = y_raw_fit[peak_idx_raw] - bl_raw[peak_idx_raw]
    
    # modpoly on empty-cell-corrected
    bl_ec = Baseline(x_data=x_ec_fit).modpoly(y_ec_fit, poly_order=3)[0]
    ec_corrected = y_ec_fit[peak_idx_ec] - bl_ec[peak_idx_ec]
    
    print(f"{file_num:6d} | {raw_I:10.6f} | {ec_I:10.6f} | {raw_corrected:18.6f} | {ec_corrected:22.6f}")

# Step 3: Check what happens in Cell 20 widget - is waxs_dict_to_correct = corrected_waxs or raw?
print("\n=== Critical check: what does the widget use? ===")
print("Cell 20 code says:")
print("  if 'waxs_dict_to_correct' not in locals():")
print("      waxs_dict_to_correct = corrected_waxs if 'corrected_waxs' in locals() else all_data['WAXS']")
print("")
print("So waxs_dict_to_correct SHOULD be corrected_waxs (empty cell corrected).")
print("BUT: if the user re-runs Cell 20 without restarting kernel,")
print("  waxs_dict_to_correct might already exist from a previous run!")
