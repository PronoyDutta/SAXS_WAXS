import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataCorrection:
    def __init__(self, data_dict):
        """
        Initialize the DataCorrection class.

        Parameters:
        - data_dict (dict): Dictionary of raw datasets (e.g., SAXS/WAXS files).
        """
        self.data_dict = data_dict

    def subtract_background(self, empty_cell_dict=None, static_background=None, capillary_df=None, start_row=0, end_row_idx=None, plot=False):
        """
        Subtract background intensity from data.
        Supports either a dictionary of empty cell files (file-by-file correction)
        or a static background file applied to all. Capillary data can be optionally added back.
        
        Parameters:
        - empty_cell_dict (dict): Dictionary of DataFrames for file-specific empty cell background.
        - static_background (pd.DataFrame): Single DataFrame for static background.
        - capillary_df (pd.DataFrame): DataFrame containing capillary intensity to add back.
        - start_row (int): Starting row index for slicing data.
        - end_row_idx (int): Ending row index (exclusive) for slicing.
        - plot (bool): Whether to plot a sample of the corrected data.
        
        Returns:
        - dict: A dictionary containing the corrected DataFrames.
        """
        result_dict = {}
        
        for file_num, df in self.data_dict.items():
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Data file with key {file_num} is not a pandas DataFrame.")
            
            _end_row = end_row_idx if end_row_idx is not None else len(df)
            
            if len(df) < _end_row:
                print(f"Skipping File {file_num}: Not enough rows (found {len(df)}, expected {_end_row})")
                continue
                
            new_df = df.iloc[start_row:_end_row, :2].copy()
            
            # Step 1: Subtract background (either empty_cell_dict or static_background)
            if empty_cell_dict is not None and file_num in empty_cell_dict:
                new_df.iloc[:, 1] = df.iloc[start_row:_end_row, 1] - empty_cell_dict[file_num].iloc[start_row:_end_row, 1]
            elif static_background is not None:
                new_df.iloc[:, 1] = df.iloc[start_row:_end_row, 1] - static_background.iloc[start_row:_end_row, 1]
            else:
                new_df.iloc[:, 1] = df.iloc[start_row:_end_row, 1] # No background subtraction if none provided
                
            # Step 2: Add back capillary intensity (if provided)
            if capillary_df is not None:
                new_df.iloc[:, 1] += capillary_df.iloc[start_row:_end_row, 1]
                
            result_dict[file_num] = new_df

        if plot and result_dict:
            plt.figure(figsize=(10, 6))
            # Plot up to 5 lines so it isn't cluttered
            plotted = 0
            for key, df in result_dict.items():
                if plotted >= 5:
                    break
                plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=f'File {key}')
                plotted += 1
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('q (nm^-1) / 2Theta (deg)')
            plt.ylabel('Corrected Intensity (a.u.)')
            plt.title('Background Subtracted Data (Sample)')
            plt.legend()
            plt.show()

        return result_dict
