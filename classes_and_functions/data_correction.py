import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataCorrection:
    def __init__(self, data_files, background_file):
        """
        Initialize the DataCorrection class.

        Parameters:
        - data_files (dict): Dictionary of raw datasets (e.g., SAXS/WAXS files).
        - background_file (pd.DataFrame): Dataset for background correction.
        """
        self.data_files = data_files
        self.background_file = background_file

    def subtract_background(self, plot=True):
        """
        Subtract the background from each dataset and optionally plot them.

        Parameters:
        - plot (bool): Whether to plot the corrected data (default is True).

        Returns:
        - dict: A dictionary containing the corrected DataFrames.
        """
        corrected_files = {}

        for key in self.data_files.keys():
            df = self.data_files[key]  # Fetch the DataFrame using the key

            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Data file with key {key} is not a pandas DataFrame.")

            # Perform background correction
            corrected_y = df.iloc[:, 1] - self.background_file.iloc[:len(df), 1]
            corrected_df = pd.DataFrame({'X': df.iloc[:, 0], 'Y_corrected': corrected_y})

            # Save corrected DataFrame
            corrected_files[key] = corrected_df

        # Optionally plot the corrected data
        if plot:
            plt.figure(figsize=(10, 6))
            for key, df in corrected_files.items():
                plt.plot(df['X'], df['Y_corrected'])
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('q (nm^-1)')
            plt.ylabel('Corrected Intensity (a.u.)')
            plt.title('Background Subtracted Data')
            plt.legend()
            plt.show()

        return corrected_files



