import pandas as pd
import h5py
import os
import glob
from galvani import BioLogic as BL
import tkinter as tk
from tkinter import filedialog, messagebox

def read_data(file_name):
    legend_temp = ""
    data_line_temp = []
    time_stamp_temp = ""
    data_storing = False
    
    with open(file_name, 'r') as file:
        for line in file:
            if 'q(A-1)' in line:
                data_storing = True
                continue
            if 'Comment' in line:
                legend_temp = line.split()[4]
            if 'Date' in line:
                time_stamp_temp = line.split()[2] 
            if data_storing:
                data_line_temp.append(line.strip())
    df_temp = pd.DataFrame([line.split() for line in data_line_temp])
    df_temp = df_temp.astype(float)

    return legend_temp, time_stamp_temp, df_temp

def read_electrochemical_data(mpr_file_path):
    mpr = BL.MPRfile(mpr_file_path)
    elec_df_temp = pd.DataFrame(mpr.data)
    return elec_df_temp

def save_to_hdf5(file_names, mpr_file_path, output_file, user_notes, sax_bg, wax_bg):
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
        
        if wax_bg:
            bg_waxs_legend, bg_waxs_timestamp, bg_waxs_data = read_data(wax_bg)
            bg_group = hdf5_file.create_group('Backgrounds/WAXS')
            bg_group.create_dataset('WAXS_background', data=bg_waxs_data.values)

        # Read and save electrochemical data if provided
        if mpr_file_path:
            elec_df_temp = read_electrochemical_data(mpr_file_path)
            elec_group = hdf5_file.create_group('ElectrochemicalData')
            elec_group.create_dataset('data', data=elec_df_temp.values)
            elec_group.create_dataset('headers', data=[header.encode('utf-8') for header in elec_df_temp.columns])
        
        # Create a group for user notes
        notes_group = hdf5_file.create_group('UserNotes')
        notes_group.create_dataset('experiment_notes', data=user_notes.encode('utf-8'))

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
            save_to_hdf5(self.file_names, self.mpr_file_path, output_file, user_notes, self.sax_bg, self.wax_bg)
            messagebox.showinfo('Success', 'Data successfully saved to HDF5 file.')

if __name__ == "__main__":
    root = tk.Tk()
    app = HDF5App(root)
    root.mainloop()
