import os
import re
import glob
import pandas as pd
from datetime import datetime
import h5py
from galvani import BioLogic as BL

def _resolve_path(path):
    if not path or pd.isna(path):
        return path
    if not os.path.isabs(path):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.normpath(os.path.join(repo_root, path))
    return path

class DataReader:
    """
    A unified class to read and process SAXS, WAXS, and electrochemical data files.
    Supports reading from a directory of individual files (.dat, .mpr) or a single HDF5 file.
    """

    def __init__(self, data_format, data_path, mpr_file_path=None):
        """
        Parameters:
        - data_format (str): 'Individual' or 'HDF5'
        - data_path (str): Path to the directory (if Individual) or the .h5 file (if HDF5)
        - mpr_file_path (str): Path to the .mpr file (only needed if data_format is 'Individual')
        """
        self.data_format = data_format.lower()
        self.data_path = _resolve_path(data_path)
        self.mpr_file_path = _resolve_path(mpr_file_path)

    def _read_individual_dat(self, file_path):
        legend_temp = None
        time_stamp_temp = None
        data_lines = []
        data_storing = False

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()

                if line.startswith("################################################################################"):
                    data_storing = True
                    continue

                if 'Comment' in line:
                    parts = line.split()
                    if len(parts) > 4:
                        legend_temp = parts[4]
                if 'Date' in line:
                    parts = line.split()
                    if len(parts) > 2:
                        time_stamp_temp = parts[2]
                    
                if data_storing and line and not line.startswith("#"):
                    data_lines.append(line)

        if len(data_lines) < 2:
            return legend_temp, time_stamp_temp, None

        df_temp = pd.DataFrame([re.split(r'\s+', line) for line in data_lines])
        df_temp.columns = df_temp.iloc[0]
        df_temp = df_temp[1:]
        df_temp = df_temp.apply(pd.to_numeric, errors='coerce')

        return legend_temp, time_stamp_temp, df_temp

    def read_data(self):
        """
        Reads the data based on the initialized format.
        
        Returns a dictionary containing:
        {
            'SAXS': {file_number: DataFrame, ...},
            'WAXS': {file_number: DataFrame, ...},
            'TimeStamps_SAXS': {file_number: timestamp, ...},
            'TimeStamps_WAXS': {file_number: timestamp, ...},
            'Legends': {file_number: legend, ...},
            'Electrochemical': DataFrame,
            'Background_SAXS': DataFrame (if available),
            'Background_WAXS': DataFrame (if available)
        }
        """
        if self.data_format == 'hdf5':
            return self._read_hdf5()
        elif self.data_format == 'individual':
            return self._read_individual()
        else:
            raise ValueError(f"Unknown data_format: {self.data_format}. Use 'HDF5' or 'Individual'.")

    def _read_hdf5(self):
        result = {
            'SAXS': {}, 'WAXS': {},
            'TimeStamps_SAXS': {}, 'TimeStamps_WAXS': {},
            'Legends': {},
            'Electrochemical': None,
            'Background_SAXS': None, 'Background_WAXS': None
        }

        with h5py.File(self.data_path, 'r') as hdf5_file:
            if 'SAXS' in hdf5_file:
                for idx, key in enumerate(hdf5_file['SAXS'].keys(), start=1):
                    result['SAXS'][idx] = pd.DataFrame(hdf5_file['SAXS'][key][:])
                    result['TimeStamps_SAXS'][idx] = hdf5_file['SAXS'][key].attrs.get('timestamp')
                    result['Legends'][idx] = f"SAXS_{idx}"

            if 'WAXS' in hdf5_file:
                for idx, key in enumerate(hdf5_file['WAXS'].keys(), start=1):
                    result['WAXS'][idx] = pd.DataFrame(hdf5_file['WAXS'][key][:])
                    result['TimeStamps_WAXS'][idx] = hdf5_file['WAXS'][key].attrs.get('timestamp')

            if 'ElectrochemicalData' in hdf5_file:
                if 'data' in hdf5_file['ElectrochemicalData']:
                    elec_data = hdf5_file['ElectrochemicalData']['data'][:]
                    if 'headers' in hdf5_file['ElectrochemicalData']:
                        headers = [h.decode('utf-8') for h in hdf5_file['ElectrochemicalData']['headers'][:]]
                        result['Electrochemical'] = pd.DataFrame(elec_data, columns=headers)
                    else:
                        result['Electrochemical'] = pd.DataFrame(elec_data)

            if 'Backgrounds' in hdf5_file:
                if 'SAXS' in hdf5_file['Backgrounds']:
                    for key in hdf5_file['Backgrounds']['SAXS'].keys():
                        result['Background_SAXS'] = pd.DataFrame(hdf5_file['Backgrounds']['SAXS'][key][:])
                if 'WAXS' in hdf5_file['Backgrounds']:
                    for key in hdf5_file['Backgrounds']['WAXS'].keys():
                        result['Background_WAXS'] = pd.DataFrame(hdf5_file['Backgrounds']['WAXS'][key][:])

        return result

    def _read_individual(self):
        result = {
            'SAXS': {}, 'WAXS': {},
            'TimeStamps_SAXS': {}, 'TimeStamps_WAXS': {},
            'Legends': {},
            'Electrochemical': None,
            'Background_SAXS': None, 'Background_WAXS': None
        }

        if not os.path.isdir(self.data_path):
            raise NotADirectoryError(f"Data path is not a directory: {self.data_path}")

        # Read SAXS files
        saxs_files = sorted(glob.glob(os.path.join(self.data_path, '*_0_*.dat')))
        for idx, file_path in enumerate(saxs_files, start=1):
            legend, ts, df = self._read_individual_dat(file_path)
            if df is not None:
                result['SAXS'][idx] = df
                result['TimeStamps_SAXS'][idx] = ts
                result['Legends'][idx] = legend

        # Read WAXS files
        waxs_files = sorted(glob.glob(os.path.join(self.data_path, '*_1_*.dat')))
        for idx, file_path in enumerate(waxs_files, start=1):
            legend, ts, df = self._read_individual_dat(file_path)
            if df is not None:
                result['WAXS'][idx] = df
                result['TimeStamps_WAXS'][idx] = ts

        # Read Electrochemical MPR file
        if self.mpr_file_path and os.path.isfile(self.mpr_file_path):
            try:
                mpr = BL.MPRfile(self.mpr_file_path)
                result['Electrochemical'] = pd.DataFrame(mpr.data)
            except Exception as e:
                print(f"Warning: Could not read MPR file: {e}")

        return result

def read_background_file(file_path):
    """
    Helper to read a single background/capillary file.
    """
    file_path = _resolve_path(file_path)
    reader = DataReader('individual', '')
    _, _, df = reader._read_individual_dat(file_path)
    return df

def read_background_directory(dir_path, pattern):
    """
    Helper to read a directory of background files matching a pattern.
    Example pattern: '*_0_*.dat' for SAXS or '*_1_*.dat' for WAXS.
    Returns a dictionary of {file_idx: DataFrame}.
    """
    dir_path = _resolve_path(dir_path)
    if not dir_path or not os.path.isdir(dir_path):
        return None
        
    result_dict = {}
    reader = DataReader('individual', '')
    files = sorted(glob.glob(os.path.join(dir_path, pattern)))
    for idx, file_path in enumerate(files, start=1):
        _, _, df = reader._read_individual_dat(file_path)
        if df is not None:
            result_dict[idx] = df
            
    return result_dict
