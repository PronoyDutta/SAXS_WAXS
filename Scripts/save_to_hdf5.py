import pandas as pd
import h5py
import os
from galvani import BioLogic as BL
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk  # Progressbar
import re

def read_data(file_name):
    legend_temp = None
    time_stamp_temp = None
    data_lines = []
    data_storing = False  # Flag to detect when to start reading data

    with open(file_name, 'r', encoding='utf-8', errors='replace') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing spaces

            # Detect the last separator line and start storing data
            if line.startswith("################################################################################"):
                data_storing = True  # Start reading data after the last separator
                continue

            # Extract metadata
            if 'Comment' in line:
                legend_temp = line.split()[4]
            if 'Date' in line:
                time_stamp_temp = line.split()[2]
                
            # Store numerical data after the separator
            if data_storing and line and not line.startswith("#"):  # Ignore empty and comment lines
                data_lines.append(line)

    # Ensure we have data before processing
    if len(data_lines) < 2:  # Must have at least one data row + header
        print("No data found in the file!")
        return legend_temp, time_stamp_temp, None

    # Convert data into a DataFrame (handle variable spaces properly)
    df_temp = pd.DataFrame([re.split(r'\s+', line) for line in data_lines])

    # Assign correct column names from the first row
    df_temp.columns = df_temp.iloc[0]  # First row contains column names
    df_temp = df_temp[1:]  # Remove the header row from data

    # Convert numeric columns to float
    df_temp = df_temp.apply(pd.to_numeric, errors='coerce')

    return legend_temp, time_stamp_temp, df_temp

def read_electrochemical_data(mpr_file_path):
    mpr = BL.MPRfile(mpr_file_path)
    elec_df_temp = pd.DataFrame(mpr.data)
    return elec_df_temp

def save_to_hdf5(file_names, mpr_file_path, output_file, user_notes, sax_bg, wax_bg, progress_callback):
    total_steps = len(file_names) + (1 if sax_bg else 0) + (1 if wax_bg else 0) + (1 if mpr_file_path else 0) + 1
    step = 0

    with h5py.File(output_file, 'w') as hdf5_file:
        timestamps = {'SAXS': [], 'WAXS': []}
        legends = {'SAXS': [], 'WAXS': []}
        
        # Process SAXS and WAXS files
        for file_name in file_names:
            legend, timestamp, data = read_data(file_name)
            
            # Determine category based on filename
            if '_0_' in file_name:
                category = 'SAXS'
            elif '_1_' in file_name:
                category = 'WAXS'
            else:
                continue
            
            # Create a group for the category if it doesn't exist
            if category not in hdf5_file:
                hdf5_file.create_group(category)
            
            # Add dataset for each file
            group = hdf5_file[category]
            dataset_name = file_name.split('/')[-1].replace('.dat', '')  # or use any other naming convention
            dataset = group.create_dataset(dataset_name, data=data.values)
            
            # Store metadata as attributes
            dataset.attrs['legend'] = legend
            dataset.attrs['timestamp'] = timestamp
            
            # Collect timestamps and legends for separate saving
            timestamps[category].append((dataset_name, timestamp))
            legends[category].append((dataset_name, legend))

            step += 1
            progress_callback(step / total_steps)

        # Create a group for timestamps
        timestamp_group = hdf5_file.create_group('Timestamps')
        for category, data in timestamps.items():
            cat_group = timestamp_group.create_group(category)
            for dataset_name, timestamp in data:
                cat_group.create_dataset(dataset_name, data=timestamp)
        
        # Create a group for legends
        legend_group = hdf5_file.create_group('Legends')
        for category, data in legends.items():
            cat_group = legend_group.create_group(category)
            for dataset_name, legend in data:
                cat_group.create_dataset(dataset_name, data=legend.encode('utf-8'))
        
        # Save SAXS and WAXS background data if provided
        if sax_bg:
            bg_saxs_legend, bg_saxs_timestamp, bg_saxs_data = read_data(sax_bg)
            bg_group = hdf5_file.create_group('Backgrounds/SAXS')
            bg_group.create_dataset('SAXS_background', data=bg_saxs_data.values)
            step += 1
            progress_callback(step / total_steps)
        
        if wax_bg:
            bg_waxs_legend, bg_waxs_timestamp, bg_waxs_data = read_data(wax_bg)
            bg_group = hdf5_file.create_group('Backgrounds/WAXS')
            bg_group.create_dataset('WAXS_background', data=bg_waxs_data.values)
            step += 1
            progress_callback(step / total_steps)

        # Read and save electrochemical data if provided
        if mpr_file_path:
            elec_df_temp = read_electrochemical_data(mpr_file_path)
            elec_group = hdf5_file.create_group('ElectrochemicalData')
            elec_group.create_dataset('data', data=elec_df_temp.values)
            elec_group.create_dataset('headers', data=[header.encode('utf-8') for header in elec_df_temp.columns])
            step += 1
            progress_callback(step / total_steps)
        
        # Create a group for user notes
        notes_group = hdf5_file.create_group('UserNotes')
        notes_group.create_dataset('experiment_notes', data=user_notes.encode('utf-8'))

        step += 1
        progress_callback(step / total_steps)

class HDF5App:
    def __init__(self, root):
        self.root = root
        self.root.title('SAXS_WAXS to HDF5')
        
        self.file_names = []
        self.mpr_file_path = ''
        self.sax_bg = ''
        self.wax_bg = ''
        
        # GUI Elements
        self.select_files_button = tk.Button(root, text='Select SAXS/WAXS Files', command=self.select_files)
        self.select_files_button.pack(pady=10)
        
        self.select_mpr_button = tk.Button(root, text='Select Electrochemical Data File (Optional)', command=self.select_mpr)
        self.select_mpr_button.pack(pady=10)
        
        self.select_saxs_bg_button = tk.Button(root, text='Select SAXS Background File (Optional)', command=self.select_saxs_bg)
        self.select_saxs_bg_button.pack(pady=10)
        
        self.select_waxs_bg_button = tk.Button(root, text='Select WAXS Background File (Optional)', command=self.select_waxs_bg)
        self.select_waxs_bg_button.pack(pady=10)
        
        self.notes_label = tk.Label(root, text='Enter detailed experimental notes:')
        self.notes_label.pack(pady=5)
        
        self.notes_text = tk.Text(root, height=5, width=40)
        self.notes_text.pack(pady=10)
        
        self.save_button = tk.Button(root, text='Save to HDF5', command=self.save_to_hdf5)
        self.save_button.pack(pady=10)

        # Progress Bar (initially hidden)
        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        
    def select_files(self):
        file_names = filedialog.askopenfilenames(filetypes=[('Data Files', '*.dat')])
        self.file_names = list(file_names)
        messagebox.showinfo('Files Selected', f'Selected {len(self.file_names)} files.')
    
    def select_mpr(self):
        self.mpr_file_path = filedialog.askopenfilename(filetypes=[('MPR Files', '*.mpr')])
        if self.mpr_file_path:
            messagebox.showinfo('MPR File Selected', f'Selected MPR file: {os.path.basename(self.mpr_file_path)}')
    
    def select_saxs_bg(self):
        self.sax_bg = filedialog.askopenfilename(filetypes=[('Data Files', '*.dat')])
        if self.sax_bg:
            messagebox.showinfo('SAXS Background Selected', f'Selected SAXS background file: {os.path.basename(self.sax_bg)}')
    
    def select_waxs_bg(self):
        self.wax_bg = filedialog.askopenfilename(filetypes=[('Data Files', '*.dat')])
        if self.wax_bg:
            messagebox.showinfo('WAXS Background Selected', f'Selected WAXS background file: {os.path.basename(self.wax_bg)}')
    
    def save_to_hdf5(self):
        if not self.file_names:
            messagebox.showerror('Error', 'Please select SAXS/WAXS files before saving.')
            return
        
        user_notes = self.notes_text.get("1.0", tk.END).strip()
        output_file = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[('HDF5 Files', '*.h5')])
        
        if output_file:
            # Show the progress bar when the 'SAVE' button is clicked
            self.progress.pack(pady=10)
            self.progress["value"] = 0
            self.progress.update()

            def update_progress(progress):
                self.progress["value"] = progress * 100
                self.progress.update()

            save_to_hdf5(self.file_names, self.mpr_file_path, output_file, user_notes, self.sax_bg, self.wax_bg, update_progress)
            messagebox.showinfo('Success', 'Data successfully saved to HDF5 file.')
            
            # Hide the progress bar after the task is complete
            self.progress.pack_forget()

if __name__ == "__main__":
    root = tk.Tk()
    app = HDF5App(root)
    root.mainloop()
