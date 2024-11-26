import matplotlib.pyplot as plt 
from read_data import DataReader as reader 

file_path = r'C:\Users\b1104371\OneDrive - Universität Salzburg\SAXS_backup\Electrolyte_variation\After_Beamtime\2503SKJB\20241120_SKJB_DOLDMEtto3_Al-windows\test'
datareader = reader(file_path)
all_data = datareader.read_all_files()
print(all_data['SAXS'].head())