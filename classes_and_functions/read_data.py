import os
import pandas as pd

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
        # Similar logic to SAXS but might include WAXS-specific parsing
        return self.read_saxs_file(file_name)  # Placeholder, modify if different

    def read_electrochemical_file(self, file_name):
        """
        Read an electrochemical data file (e.g., MPR files).

        Parameters:
        file_name (str): The name of the electrochemical file to read.

        Returns:
        tuple: (DataFrame, headers) - The electrochemical data and associated headers.
        """
        from galvani import BioLogic as BL
        file_path = os.path.join(self.directory, file_name)
        try:
            ec_data = BL.MPRfile(file_path)
            data = pd.DataFrame(ec_data.data)
            headers = ec_data.headers
            return data, headers
        except Exception as e:
            raise ValueError(f"Error reading electrochemical file '{file_name}': {e}")

    def read_file(self, file_name):
        """
        General-purpose method to read a file based on its type.

        Parameters:
        file_name (str): The name of the file to read.

        Returns:
        DataFrame or tuple: Parsed data, format depends on file type.
        """
        if file_name.endswith(".dat") or file_name.endswith(".txt"):
            if "_0_" in file_name:  # Assuming '_0_' indicates SAXS
                return self.read_saxs_file(file_name)
            elif "_1_" in file_name:  # Assuming '_1_' indicates WAXS
                return self.read_waxs_file(file_name)
        elif file_name.endswith(".mpr"):
            return self.read_electrochemical_file(file_name)
        else:
            raise ValueError(f"Unsupported file type for file '{file_name}'.")

    def read_all_files(self):
        """
        Read all files in the directory and categorize by type.

        Returns:
        dict: Dictionary containing data categorized by type:
              {'SAXS': [...], 'WAXS': [...], 'Electrochemical': [...]}
        """
        all_data = {'SAXS': {}, 'WAXS': {}, 'Electrochemical': {}}
        for file_name in self.list_files():
            try:
                if "_0_" in file_name:
                    all_data['SAXS'][file_name] = self.read_saxs_file(file_name)
                elif "_1_" in file_name:
                    all_data['WAXS'][file_name] = self.read_waxs_file(file_name)
                elif file_name.endswith(".mpr"):
                    all_data['Electrochemical'][file_name] = self.read_electrochemical_file(file_name)
            except Exception as e:
                print(f"Warning: Could not read file '{file_name}': {e}")
        return all_data
