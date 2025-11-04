import os
import pandas as pd
from datetime import datetime
from galvani import BioLogic as BL  # For electrochemical data processing

class DataReader:
    """
    A class to read and process SAXS, WAXS, and electrochemical data files.
    """

    def __init__(self, directory):
        """
        Initialize the DataReader with a directory containing data files.

        Parameters:
        directory (str): Path to the directory containing data files.
        """
        self.directory = directory

    def list_files(self, file_type=None):
        """
        List all files in the directory with optional filtering by file type.

        Parameters:
        file_type (str): File extension to filter by (e.g., 'csv', 'dat', 'mpr').

        Returns:
        list: List of file names matching the criteria.
        """
        try:
            files = [f for f in os.listdir(self.directory) if os.path.isfile(os.path.join(self.directory, f))]
            if file_type:
                files = [f for f in files if f.endswith(f".{file_type}")]
            return files
        except Exception as e:
            raise FileNotFoundError(f"Error accessing directory '{self.directory}': {e}")

    def read_saxs_file(self, file_name):
        """
        Read a SAXS data file.

        Parameters:
        file_name (str): The name of the SAXS file to read.

        Returns:
        DataFrame: DataFrame containing the SAXS data.
        """
        file_path = os.path.join(self.directory, file_name)
        try:
            with open(file_path, 'r') as file:
                data_lines = []
                data_storing = False
                for line in file:
                    if 'q(A-1)' in line:
                        data_storing = True
                        continue
                    if data_storing:
                        data_lines.append(line.strip())
            data = pd.DataFrame([list(map(float, line.split())) for line in data_lines])
            return data
        except Exception as e:
            raise ValueError(f"Error reading SAXS file '{file_name}': {e}")

    def read_waxs_file(self, file_name):
        """
        Read a WAXS data file.

        Parameters:
        file_name (str): The name of the WAXS file to read.

        Returns:
        DataFrame: DataFrame containing the WAXS data.
        """
        file_path = os.path.join(self.directory, file_name)
        try:
            with open(file_path, 'r') as file:
                data_lines = []
                data_storing = False
                for line in file:
                    if 'q(A-1)' in line:
                        data_storing = True
                        continue
                    if data_storing:
                        data_lines.append(line.strip())
            data = pd.DataFrame([list(map(float, line.split())) for line in data_lines])
            return data
        except Exception as e:
            raise ValueError(f"Error reading WAXS file '{file_name}': {e}")

    def read_electrochemical_file(self, file_name):
        """
        Read an electrochemical data file (e.g., MPR files).

        Parameters:
        file_name (str): The name of the electrochemical file to read.

        Returns:
        DataFrame: DataFrame containing the electrochemical data.
        """
        file_path = os.path.join(self.directory, file_name)
        try:
            ec_data = BL.MPRfile(file_path)
            data = pd.DataFrame(ec_data.data)
            return data
        except Exception as e:
            raise ValueError(f"Error reading electrochemical file '{file_name}': {e}")

    def read_all_files(self):
        """
        Read all files in the directory and organize them into categories.

        Returns:
        dict: A dictionary with keys 'SAXS', 'WAXS', 'Electrochemical', and 'Timestamps',
              where:
              - 'SAXS': DataFrames for SAXS files.
              - 'WAXS': DataFrames for WAXS files.
              - 'Electrochemical': DataFrames for electrochemical files.
              - 'Timestamps': List of SAXS file timestamps.
        """
        all_data = {'SAXS': {}, 'WAXS': {}, 'Electrochemical': {}, 'Timestamps': []}
        sax_count, wax_count, ec_count = 0, 0, 0

        for file_name in self.list_files():
            try:
                if "_0_" in file_name and file_name.endswith(".dat"):  # SAXS files
                    # Extract timestamp
                    file_path = os.path.join(self.directory, file_name)
                    with open(file_path, 'r') as file:
                        timestamp = None
                        for line in file:
                            if 'Date' in line:
                                timestamp = datetime.strptime(line.split()[2], '%Y-%m-%dT%H:%M:%S')
                                break
                    # Read SAXS data
                    data = self.read_saxs_file(file_name)
                    all_data['SAXS'][sax_count] = data
                    all_data['Timestamps'].append(timestamp)
                    sax_count += 1
                elif "_1_" in file_name and file_name.endswith(".dat"):  # WAXS files
                    data = self.read_waxs_file(file_name)
                    all_data['WAXS'][wax_count] = data
                    wax_count += 1
                elif file_name.endswith(".mpr"):  # Electrochemical files
                    data = self.read_electrochemical_file(file_name)
                    all_data['Electrochemical'][ec_count] = data
                    ec_count += 1
            except Exception as e:
                print(f"Warning: Could not read file '{file_name}': {e}")

        return all_data

