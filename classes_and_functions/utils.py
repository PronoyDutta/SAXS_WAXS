import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter
import tkinter as tk
from tkinter import filedialog
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def smooth_data(data, window_length, polyorder):
    return savgol_filter(data, window_length, polyorder)

def find_nearest_time(target_time, time_dict):
    """
    Finds the nearest time in time_dict to the target_time.
    time_dict format depends on data source (HDF5 or Individual files).
    If it's from individual files, it's string format '%Y-%m-%dT%H:%M:%S'.
    If from HDF5, it might be string or seconds.
    """
    new_time_dict = {}
    keys = list(time_dict.keys())
    
    # Try parsing as datetime if it's a string, else assume it's seconds or similar
    try:
        base_time = datetime.strptime(str(time_dict[keys[0]]), '%Y-%m-%dT%H:%M:%S')
        for key in time_dict:
            current_time = datetime.strptime(str(time_dict[key]), '%Y-%m-%dT%H:%M:%S')
            time_difference = (current_time - base_time).total_seconds()
            new_time_dict[time_difference] = key
    except ValueError:
        # Fallback if timestamps are not standard strings
        base_time = float(time_dict[keys[0]])
        for key in time_dict:
            time_difference = float(time_dict[key]) - base_time
            new_time_dict[time_difference] = key
            
    nearest_time_difference = min(new_time_dict.keys(), key=lambda x: abs(x - target_time))
    nearest_key = new_time_dict[nearest_time_difference]
    
    return nearest_key, new_time_dict

def contourplot(
    plot_type, data_normalization, data_dictionary, new_time_dict,
    min_limit, max_limit, cmap_min=None, cmap_max=None,
    Normalization_file=None, save_plot=False, filename="contour_plot.png",
    decimal_points=2, scientific=False, y_axis_mode='time',
    inset_colorbar_inside=True, colorbar_position='bottom',
    tick_label_color='black', tick_label_fontsize=12,
    colorbar_tick_size=5, colorbar_tick_label_fontsize=10,
    starting_file=1, last_file=None,
    applied_current=None, active_material_mass=None,
    starting_cycle=None, final_cycle=None, ec_results=None, discharge_files=None
):
    """
    Generates a contour plot for SAXS/WAXS data correlated with electrochemical data.
    
    Parameters:
    - plot_type (str): Type of plot, options are 'saxs' or 'waxs'.
    - data_normalization (str): Normalization method, options are 'absolute' or 'relative'.
    - data_dictionary (dict): Dictionary mapping file numbers to data DataFrames.
    - new_time_dict (dict): Dictionary mapping file numbers to timestamps.
    - min_limit (float): Minimum limit for the x-axis (q or 2θ).
    - max_limit (float): Maximum limit for the x-axis (q or 2θ).
    - cmap_min (float, optional): Minimum value for the color scale. Default is None (auto).
    - cmap_max (float, optional): Maximum value for the color scale. Default is None (auto).
    - Normalization_file (int, optional): The file index to normalize against when data_normalization='relative' (e.g., discharge_file or starting_file).
    - save_plot (bool): Whether to save the plot to a file, options are True or False.
    - filename (str): Default filename to save the plot as.
    - decimal_points (int): Number of decimal points for the colorbar ticks.
    - scientific (bool): Whether to use scientific notation for the colorbar, options are True or False.
    - y_axis_mode (str): The value to plot on the Y-axis, options are 'time' or 'capacity'.
    - inset_colorbar_inside (bool): Whether to draw the side colorbar inside the plot area, options are True or False.
    - colorbar_position (str): Position of the colorbar, options are 'bottom' or 'side'.
    - tick_label_color (str): Color of the colorbar tick labels (e.g., 'black', 'white').
    - tick_label_fontsize (int): Font size for the colorbar labels.
    - colorbar_tick_size (int): Length of the colorbar ticks.
    - colorbar_tick_label_fontsize (int): Font size for the colorbar tick text.
    - starting_file (int): The starting file index to begin plotting from.
    - last_file (int, optional): The ending file index to plot up to. Default is None (uses max index).
    - elec_df_2 (DataFrame, optional): Filtered electrochemical dataframe for the left-side subplot.
    - applied_current (float, optional): Applied current in mA (required if y_axis_mode='capacity').
    - active_material_mass (float, optional): Active material mass in mg (required if y_axis_mode='capacity').
    """
    if ec_results is not None and starting_cycle is not None and final_cycle is not None:
        saxs_files = ec_results.get('saxs_file_numbers', [])
        if saxs_files and len(saxs_files) > 0:
            start_idx = max(0, int(2 * starting_cycle - 2))
            end_idx = min(len(saxs_files) - 1, int(2 * final_cycle))
            if start_idx < len(saxs_files) and end_idx < len(saxs_files):
                starting_file = saxs_files[start_idx]
                last_file = saxs_files[end_idx]
                
            if 'cycle_change_indices' in ec_results:
                cycle_indices = ec_results['cycle_change_indices']
                s_idx = None
                e_idx = None
                for i, file in enumerate(saxs_files):
                    if file >= starting_file and s_idx is None:
                        s_idx = i
                    if file >= last_file and e_idx is None:
                        e_idx = i
                        break
                if s_idx is None: s_idx = 0
                if e_idx is None or e_idx >= len(cycle_indices): e_idx = len(cycle_indices) - 1
                
                elec_df_full = ec_results.get('filtered_dataframe')
                if elec_df_full is not None:
                    try:
                        elec_df_to_plot = elec_df_full.loc[cycle_indices[s_idx]:cycle_indices[e_idx]]
                    except KeyError:
                        elec_df_to_plot = elec_df_full
                else:
                    elec_df_to_plot = None
            else:
                elec_df_to_plot = ec_results.get('filtered_dataframe') if ec_results else None
    else:
        elec_df_to_plot = ec_results.get('filtered_dataframe') if ec_results else None
        print(f"Plotting Graph for cycles {starting_cycle} to {final_cycle}")

    if last_file is None:
        last_file = max(data_dictionary.keys())

    if discharge_files is None and ec_results is not None:
        discharge_files = ec_results.get('discharge_saxs_files')

    discharge_file = "N/A"
    if discharge_files is not None and starting_cycle is not None:
        idx = int(starting_cycle) - 1
        if 0 <= idx < len(discharge_files):
            discharge_file = discharge_files[idx]
            
    if str(Normalization_file).lower() == 'starting_file':
        Normalization_file = starting_file
    elif str(Normalization_file).lower() == 'discharge_file' and discharge_file != "N/A":
        Normalization_file = discharge_file
    elif str(Normalization_file).lower() == 'last_file':
        Normalization_file = last_file

    print(f"Starting file: {starting_file}, Discharge file: {discharge_file}, Last file: {last_file}")
        
    wavelength_nm = 0.154  # Cu K-alpha wavelength in nm
    fntsize = 16
    plt.style.use('default')

    keys = list(new_time_dict.keys())
    time_t1 = []
    Intensity_list = []

    # Get X axis data (Q or 2Theta)
    X1_data_temp = 10 * data_dictionary[starting_file].iloc[:, 0]
    if plot_type == 'waxs':
        X1_data_temp = 2 * np.degrees(np.arcsin((wavelength_nm * X1_data_temp) / (4 * np.pi)))
        x_label = '2θ (°)'
    else:
        x_label = 'q (nm$^{-1}$)'

    # Filter by limit
    indices_within_range = np.where((X1_data_temp >= min_limit) & (X1_data_temp <= max_limit))
    X1_data_filtered_positions = X1_data_temp.iloc[indices_within_range]
    start_row = np.min(indices_within_range)
    end_row = np.max(indices_within_range)
    
    # ---- Robust Timestamp Extraction ----
    keys = list(new_time_dict.keys())
    if len(keys) > 0 and isinstance(new_time_dict[keys[0]], (int, np.integer, float)):
        file_vals = list(new_time_dict.values())
        if all(isinstance(v, (int, np.integer)) for v in file_vals) and min(file_vals) >= 1:
            file_to_time = {v: float(k) for k, v in new_time_dict.items()}
            base_time = file_to_time.get(starting_file, 0.0)
            get_elapsed = lambda f: file_to_time.get(f, 0.0) - base_time
        else:
            def parse_ts(v):
                if isinstance(v, (int, float, np.number)): return float(v)
                if isinstance(v, bytes): v = v.decode('utf-8')
                try: return datetime.strptime(str(v), '%Y-%m-%dT%H:%M:%S').timestamp()
                except: return float(v)
            def get_val(d, k):
                if k in d: return d[k]
                if str(k) in d: return d[str(k)]
                try: return d[int(k)]
                except: return None
            
            base_val = get_val(new_time_dict, starting_file)
            base_time = parse_ts(base_val) if base_val is not None else 0.0
            get_elapsed = lambda f: parse_ts(get_val(new_time_dict, f)) - base_time if get_val(new_time_dict, f) is not None else 0.0
    else:
        def parse_ts(v):
            if isinstance(v, (int, float, np.number)): return float(v)
            if isinstance(v, bytes): v = v.decode('utf-8')
            try: return datetime.strptime(str(v), '%Y-%m-%dT%H:%M:%S').timestamp()
            except: return float(v)
        base_val = new_time_dict.get(starting_file)
        base_time = parse_ts(base_val) if base_val is not None else 0.0
        get_elapsed = lambda f: parse_ts(new_time_dict.get(f)) - base_time if new_time_dict.get(f) is not None else 0.0

    # Process data matrices
    for file_number in range(starting_file, last_file + 1):
        if data_normalization == 'relative' and Normalization_file is not None:
            norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1] / data_dictionary[Normalization_file].iloc[start_row:end_row + 1, 1]
        else:
            norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1]

        Intensity_list.append(norm_intensity.to_numpy())
        time_t1.append(get_elapsed(file_number))

    time_hours = np.array(time_t1) / 3600
    if y_axis_mode == 'capacity' and applied_current and active_material_mass:
        y_data = (np.array(time_t1) * applied_current) / (active_material_mass * 3.6)
        y_label = 'Specific Capacity (mAh g$^{-1}$)'
    else:
        y_data = time_hours
        y_label = 'Time (h)'

    normIntensity_array = np.array(Intensity_list)
    normIntensity_array = gaussian_filter(normIntensity_array, sigma=1)

    X, Y = np.meshgrid(X1_data_filtered_positions, y_data)
    cmap1 = plt.colormaps.get_cmap('viridis')

    fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True, gridspec_kw={'width_ratios': [1, 4]})

    # Left subplot: Electrochemical
    if elec_df_to_plot is not None and not elec_df_to_plot.empty:
        # Assume 'time/s' and 'Ewe/V' are columns
        if 'time/s' in elec_df_to_plot.columns and 'Ewe/V' in elec_df_to_plot.columns:
            time_col = elec_df_to_plot['time/s']
            volt_col = elec_df_to_plot['Ewe/V']
        elif len(elec_df_to_plot.columns) > 6: # Standard bio-logic column index fallback
            time_col = elec_df_to_plot.iloc[:, 2] if 'time/s' not in elec_df_to_plot.columns else elec_df_to_plot['time/s']
            volt_col = elec_df_to_plot.iloc[:, 6] if 'Ewe/V' not in elec_df_to_plot.columns else elec_df_to_plot['Ewe/V']
        else:
            time_col = elec_df_to_plot.iloc[:, 0]
            volt_col = elec_df_to_plot.iloc[:, 1]
            
        if y_axis_mode == 'capacity' and applied_current and active_material_mass:
            left_y = ((time_col - time_col.iloc[0]) * applied_current) / (active_material_mass * 3.6)
        else:
            left_y = (time_col - time_col.iloc[0]) / 3600

        axs[0].plot(volt_col, left_y, color='r')
        axs[0].set_ylim(left_y.min(), left_y.max())
    
    axs[0].invert_xaxis()
    axs[0].set_ylabel(y_label, fontsize=fntsize)
    axs[0].set_xlabel('Potential (V)', fontsize=fntsize)

    # Right subplot: Contour
    contour = axs[1].contourf(X, Y, normIntensity_array, cmap=cmap1, vmin=cmap_min, vmax=cmap_max)

    if colorbar_position == 'side' and inset_colorbar_inside:
        inset_ax = inset_axes(axs[1], width="4%", height="50%", loc='upper right',
                              bbox_to_anchor=(0.2, 0.0, 1, 1), bbox_transform=axs[1].transAxes, borderpad=0.5)
        cbar = fig.colorbar(contour, cax=inset_ax)
        cbar.ax.tick_params(direction='out', length=colorbar_tick_size, labelsize=colorbar_tick_label_fontsize,
                            labelleft=True, labelright=False, right=False, left=True,
                            color=tick_label_color, labelcolor=tick_label_color)
    elif colorbar_position == 'side':
        cbar = fig.colorbar(contour, ax=axs[1])
        cbar.ax.tick_params(direction='out', length=colorbar_tick_size, labelsize=tick_label_fontsize,
                            labelleft=False, labelright=True, right=True, left=False)
    else:
        cbar_ax = fig.add_axes([0.3, 0.08, 0.55, 0.03])
        cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(direction='out', length=colorbar_tick_size, labelsize=tick_label_fontsize)

    # Label
    label_text = 'rel. Intensity' if data_normalization == 'relative' else 'Intensity (a.u.)'
    if colorbar_position == 'side' and inset_colorbar_inside:
        axs[1].text(1.2, 1.01, label_text, transform=axs[1].transAxes, fontsize=tick_label_fontsize, ha='right', va='bottom', color='black')
    else:
        cbar.set_label(label_text, fontsize=fntsize)
    
    # Tick values
    cbar_min = cmap_min if cmap_min is not None else normIntensity_array.min()
    cbar_max = cmap_max if cmap_max is not None else normIntensity_array.max()
    cbar_mid = (cbar_min + cbar_max) / 2
    cbar.set_ticks([cbar_min, cbar_mid, cbar_max])

    if scientific:
        factor = 1e4
        cbar.set_ticklabels([f"{cbar_min*factor:.{decimal_points}f}", f"{cbar_mid*factor:.{decimal_points}f}", f"{cbar_max*factor:.{decimal_points}f}"])
    else:
        cbar.set_ticklabels([f"{cbar_min:.{decimal_points}f}", f"{cbar_mid:.{decimal_points}f}", f"{cbar_max:.{decimal_points}f}"])

    if plot_type == 'saxs':
        axs[1].set_xscale('log')
        axs[1].xaxis.set_major_formatter(mticker.ScalarFormatter())
        ticks = [min_limit] + list(range(int(np.ceil(min_limit)), int(max_limit) + 1, 2))
        axs[1].set_xticks(ticks)
        axs[1].set_xticklabels([f'{tick:.1f}' for tick in ticks], fontsize=fntsize)
        axs[1].xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1, numticks=10))

    axs[1].set_xlabel(x_label, fontsize=fntsize)
    axs[0].tick_params(axis='both', labelsize=fntsize)
    axs[1].tick_params(axis='both', labelsize=fntsize)
    fig.subplots_adjust(wspace=0.05, left=0.15, right=0.85, top=0.85, bottom=0.2)

    if save_plot:
        try:
            root = tk.Tk()
            root.withdraw()
            save_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=filename, filetypes=[("PNG files", "*.png")])
            if save_path:
                plt.savefig(save_path, dpi=600)
                print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Could not open save dialog, saving to current directory as {filename}")
            plt.savefig(filename, dpi=600)

    plt.show()

def subtract_background(data_dict, background_df, align_target_X=None):
    """
    Subtracts a background DataFrame from a dictionary of data DataFrames.
    If align_target_X is provided (e.g., 3.4 for WAXS), it interpolates the 
    background to align with the data at that specific target X value before subtracting.
    """
    if background_df is None or background_df.empty:
        print("No background data provided.")
        return data_dict
        
    corrected_dict = {}
    for key in data_dict:
        X_data = data_dict[key].iloc[:, 0]
        Y_data = data_dict[key].iloc[:, 1]
        
        if align_target_X is not None:
            # Interpolate to align background vertically with the signal
            if X_data.min() <= align_target_X <= X_data.max():
                Y_value_at_target = np.interp(align_target_X, X_data, Y_data)
                background_Y_value_at_target = np.interp(align_target_X, background_df.iloc[:, 0], background_df.iloc[:, 1])
                offset = Y_value_at_target - background_Y_value_at_target
                aligned_background_Y = background_df.iloc[:, 1] + offset
                modified_Y = Y_data - aligned_background_Y
            else:
                print(f"Target X {align_target_X} is out of bounds for file {key}")
                modified_Y = Y_data - background_df.iloc[:len(Y_data), 1].values
        else:
            # Direct subtraction
            end_line = min(len(Y_data), len(background_df))
            modified_Y = Y_data.iloc[:end_line] - background_df.iloc[:end_line, 1].values
            X_data = X_data.iloc[:end_line]
            
        df = pd.DataFrame({0: X_data.values, 1: modified_Y.values})
        corrected_dict[key] = df
        
    return corrected_dict

def combined_contourplot(
    saxs_data_normalization, waxs_data_normalization, saxs_dict, waxs_dict, new_time_dict,
    saxs_min_limit, saxs_max_limit, waxs_min_limit, waxs_max_limit,
    saxs_cmap_min=None, saxs_cmap_max=None, waxs_cmap_min=None, waxs_cmap_max=None,
    Normalization_file_saxs=None, Normalization_file_waxs=None,
    save_plot=False, filename="combined_contour_plot.png",
    decimal_points=2, scientific=False, y_axis_mode='time',
    colorbar_position='bottom', inset_colorbar_inside=True,
    tick_label_color='black', tick_label_fontsize=12,
    colorbar_tick_size=5, colorbar_tick_label_fontsize=10,
    starting_file=1, last_file=None,
    applied_current=None, active_material_mass=None,
    discharge_files=None,
    starting_cycle=None, final_cycle=None, ec_results=None
):
    """
    Generates a combined contour plot for Electrochemical data, SAXS, and WAXS side-by-side.
    """
    if ec_results is not None and starting_cycle is not None and final_cycle is not None:
        saxs_files = ec_results.get('saxs_file_numbers', [])
        if saxs_files and len(saxs_files) > 0:
            start_idx = max(0, int(2 * starting_cycle - 2))
            end_idx = min(len(saxs_files) - 1, int(2 * final_cycle))
            if start_idx < len(saxs_files) and end_idx < len(saxs_files):
                starting_file = saxs_files[start_idx]
                last_file = saxs_files[end_idx]
                
            # Filter the electrochemistry dataframe to only the selected cycles
            if 'cycle_change_indices' in ec_results:
                cycle_indices = ec_results['cycle_change_indices']
                s_idx = None
                e_idx = None
                for i, file in enumerate(saxs_files):
                    if file >= starting_file and s_idx is None:
                        s_idx = i
                    if file >= last_file and e_idx is None:
                        e_idx = i
                        break
                if s_idx is None: s_idx = 0
                if e_idx is None or e_idx >= len(cycle_indices): e_idx = len(cycle_indices) - 1
                
                elec_df_full = ec_results.get('filtered_dataframe')
                if elec_df_full is not None:
                    try:
                        elec_df_to_plot = elec_df_full.loc[cycle_indices[s_idx]:cycle_indices[e_idx]]
                    except KeyError:
                        elec_df_to_plot = elec_df_full
                else:
                    elec_df_to_plot = None
            else:
                elec_df_to_plot = ec_results.get('filtered_dataframe') if ec_results else None
    else:
        elec_df_to_plot = ec_results.get('filtered_dataframe') if ec_results else None
        print(f"Plotting Combined Graph for cycles {starting_cycle} to {final_cycle}")

    if last_file is None:
        last_file = max(saxs_dict.keys())
        
    discharge_file = "N/A"
    if discharge_files is not None and starting_cycle is not None:
        idx = int(starting_cycle) - 1
        if 0 <= idx < len(discharge_files):
            discharge_file = discharge_files[idx]
            
    if str(Normalization_file_saxs).lower() == 'starting_file':
        Normalization_file_saxs = starting_file
    elif str(Normalization_file_saxs).lower() == 'discharge_file' and discharge_file != "N/A":
        Normalization_file_saxs = discharge_file
        
    if str(Normalization_file_waxs).lower() == 'starting_file':
        Normalization_file_waxs = starting_file
    elif str(Normalization_file_waxs).lower() == 'discharge_file' and discharge_file != "N/A":
        Normalization_file_waxs = discharge_file

    print(f"Starting file: {starting_file}, Discharge file: {discharge_file}, Last file: {last_file}")
        
    wavelength_nm = 0.154  # Cu K-alpha wavelength in nm
    fntsize = 16
    plt.style.use('default')

    keys = list(new_time_dict.keys())
    time_t1 = []
    
    def get_grid(data_dictionary, min_limit, max_limit, normalization_type, is_waxs=False, norm_file=None):
        X1_data_temp = 10 * data_dictionary[starting_file].iloc[:, 0]
        if is_waxs:
            X1_data_temp = 2 * np.degrees(np.arcsin((wavelength_nm * X1_data_temp) / (4 * np.pi)))
            
        indices = np.where((X1_data_temp >= min_limit) & (X1_data_temp <= max_limit))
        X1_filtered = X1_data_temp.iloc[indices]
        start_row, end_row = np.min(indices), np.max(indices)
        
        Intensity_list = []
        for file_number in range(starting_file, last_file + 1):
            if normalization_type == 'relative' and norm_file is not None:
                norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1] / data_dictionary[norm_file].iloc[start_row:end_row + 1, 1]
            else:
                norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1]
            Intensity_list.append(norm_intensity.to_numpy())
            
        normIntensity_array = np.array(Intensity_list)
        return X1_filtered, gaussian_filter(normIntensity_array, sigma=1)

    X_saxs, Z_saxs = get_grid(saxs_dict, saxs_min_limit, saxs_max_limit, saxs_data_normalization, is_waxs=False, norm_file=Normalization_file_saxs)
    X_waxs, Z_waxs = get_grid(waxs_dict, waxs_min_limit, waxs_max_limit, waxs_data_normalization, is_waxs=True, norm_file=Normalization_file_waxs)
    

    # ---- Robust Timestamp Extraction ----
    keys = list(new_time_dict.keys())
    if len(keys) > 0 and isinstance(new_time_dict[keys[0]], (int, np.integer, float)):
        file_vals = list(new_time_dict.values())
        if all(isinstance(v, (int, np.integer)) for v in file_vals) and min(file_vals) >= 1:
            file_to_time = {v: float(k) for k, v in new_time_dict.items()}
            base_time = file_to_time.get(starting_file, 0.0)
            get_elapsed = lambda f: file_to_time.get(f, 0.0) - base_time
        else:
            def parse_ts(v):
                if isinstance(v, (int, float, np.number)): return float(v)
                if isinstance(v, bytes): v = v.decode('utf-8')
                try: return datetime.strptime(str(v), '%Y-%m-%dT%H:%M:%S').timestamp()
                except: return float(v)
            def get_val(d, k):
                if k in d: return d[k]
                if str(k) in d: return d[str(k)]
                try: return d[int(k)]
                except: return None
            
            base_val = get_val(new_time_dict, starting_file)
            base_time = parse_ts(base_val) if base_val is not None else 0.0
            get_elapsed = lambda f: parse_ts(get_val(new_time_dict, f)) - base_time if get_val(new_time_dict, f) is not None else 0.0
    else:
        def parse_ts(v):
            if isinstance(v, (int, float, np.number)): return float(v)
            if isinstance(v, bytes): v = v.decode('utf-8')
            try: return datetime.strptime(str(v), '%Y-%m-%dT%H:%M:%S').timestamp()
            except: return float(v)
        base_val = new_time_dict.get(starting_file)
        base_time = parse_ts(base_val) if base_val is not None else 0.0
        get_elapsed = lambda f: parse_ts(new_time_dict.get(f)) - base_time if new_time_dict.get(f) is not None else 0.0

    for file_number in range(starting_file, last_file + 1):
        time_t1.append(get_elapsed(file_number))

    if y_axis_mode == 'capacity' and applied_current and active_material_mass:
        y_data = (np.array(time_t1) * applied_current) / (active_material_mass * 3.6)
        y_label = 'Specific Capacity (mAh g$^{-1}$)'
    else:
        y_data = np.array(time_t1) / 3600
        y_label = 'Time (h)'

    X1, Y1 = np.meshgrid(X_saxs, y_data)
    X2, Y2 = np.meshgrid(X_waxs, y_data)
    
    cmap1 = plt.colormaps.get_cmap('viridis')

    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, gridspec_kw={'width_ratios': [1.5, 3, 3]})

    if elec_df_to_plot is not None and not elec_df_to_plot.empty:
        if 'time/s' in elec_df_to_plot.columns and 'Ewe/V' in elec_df_to_plot.columns:
            time_col, volt_col = elec_df_to_plot['time/s'], elec_df_to_plot['Ewe/V']
        elif len(elec_df_to_plot.columns) > 6:
            time_col = elec_df_to_plot.iloc[:, 2] if 'time/s' not in elec_df_to_plot.columns else elec_df_to_plot['time/s']
            volt_col = elec_df_to_plot.iloc[:, 6] if 'Ewe/V' not in elec_df_to_plot.columns else elec_df_to_plot['Ewe/V']
        else:
            time_col, volt_col = elec_df_to_plot.iloc[:, 0], elec_df_to_plot.iloc[:, 1]
            
        if y_axis_mode == 'capacity' and applied_current and active_material_mass:
            left_y = ((time_col - time_col.iloc[0]) * applied_current) / (active_material_mass * 3.6)
        else:
            left_y = (time_col - time_col.iloc[0]) / 3600
        axs[0].plot(volt_col, left_y, color='r')
        axs[0].set_ylim(left_y.min(), left_y.max())
    
    axs[0].invert_xaxis()
    axs[0].set_ylabel(y_label, fontsize=fntsize)
    axs[0].set_xlabel('Potential (V)', fontsize=fntsize)

    axs[1].set_title('SAXS', fontsize=fntsize, fontweight='bold', loc='center')
    contour_saxs = axs[1].contourf(X1, Y1, Z_saxs, cmap=cmap1, vmin=saxs_cmap_min, vmax=saxs_cmap_max)
    axs[1].set_xlabel('q (nm$^{-1}$)', fontsize=fntsize)
    axs[1].set_xscale('log')
    axs[1].xaxis.set_major_formatter(mticker.ScalarFormatter())
    ticks = [saxs_min_limit] + list(range(int(np.ceil(saxs_min_limit)), int(saxs_max_limit) + 1, 2))
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels([f'{tick:.1f}' for tick in ticks], fontsize=fntsize)

    axs[2].set_title('WAXS', fontsize=fntsize, fontweight='bold', loc='center')
    contour_waxs = axs[2].contourf(X2, Y2, Z_waxs, cmap=cmap1, vmin=waxs_cmap_min, vmax=waxs_cmap_max)
    axs[2].set_xlabel('2θ (°)', fontsize=fntsize)

    label_saxs = 'rel. Intensity' if saxs_data_normalization == 'relative' else 'Intensity (a.u.)'
    label_waxs = 'rel. Intensity' if waxs_data_normalization == 'relative' else 'Intensity (a.u.)'
    
    def setup_cbar(cbar, cbar_min, cbar_max, is_side, ax_idx, is_inset, label_text):
        cbar_min = cbar_min if cbar_min is not None else (Z_saxs.min() if ax_idx==1 else Z_waxs.min())
        cbar_max = cbar_max if cbar_max is not None else (Z_saxs.max() if ax_idx==1 else Z_waxs.max())
        cbar_mid = (cbar_min + cbar_max) / 2
        cbar.set_ticks([cbar_min, cbar_mid, cbar_max])
        if scientific:
            factor = 1e4
            cbar.set_ticklabels([f"{cbar_min*factor:.{decimal_points}f}", f"{cbar_mid*factor:.{decimal_points}f}", f"{cbar_max*factor:.{decimal_points}f}"])
        else:
            cbar.set_ticklabels([f"{cbar_min:.{decimal_points}f}", f"{cbar_mid:.{decimal_points}f}", f"{cbar_max:.{decimal_points}f}"])
            
        if is_side:
            cbar.ax.tick_params(direction='out', length=colorbar_tick_size, labelsize=colorbar_tick_label_fontsize,
                                color=tick_label_color, labelcolor=tick_label_color)
            if is_inset:
                cbar.ax.set_title(label_text, fontsize=colorbar_tick_label_fontsize, color='black', pad=10)
        else:
            cbar.ax.tick_params(direction='out', length=colorbar_tick_size, labelsize=tick_label_fontsize)
            cbar.set_label(label_text, fontsize=fntsize)

    if colorbar_position == 'side' and inset_colorbar_inside:
        for ax_idx, contour, cmin, cmax, label_text in [(1, contour_saxs, saxs_cmap_min, saxs_cmap_max, label_saxs), (2, contour_waxs, waxs_cmap_min, waxs_cmap_max, label_waxs)]:
            # Anchor OUTSIDE the plot to the right, pushing it a bit further to make room for left-ticks
            inset_ax = inset_axes(axs[ax_idx], width="5%", height="40%", loc='lower left', bbox_to_anchor=(1.15, 0.6, 1, 1), bbox_transform=axs[ax_idx].transAxes, borderpad=0)
            cb = fig.colorbar(contour, cax=inset_ax)
            cb.ax.tick_params(labelleft=True, labelright=False, right=False, left=True)
            setup_cbar(cb, cmin, cmax, True, ax_idx, True, label_text)
    elif colorbar_position == 'side':
        for ax_idx, contour, cmin, cmax, label_text in [(1, contour_saxs, saxs_cmap_min, saxs_cmap_max, label_saxs), (2, contour_waxs, waxs_cmap_min, waxs_cmap_max, label_waxs)]:
            cb = fig.colorbar(contour, ax=axs[ax_idx])
            cb.ax.tick_params(labelleft=True, labelright=False, right=False, left=True)
            setup_cbar(cb, cmin, cmax, True, ax_idx, False, label_text)
    else:
        cbar_ax1 = fig.add_axes([0.38, 0.05, 0.22, 0.03])
        cb1 = fig.colorbar(contour_saxs, cax=cbar_ax1, orientation='horizontal')
        setup_cbar(cb1, saxs_cmap_min, saxs_cmap_max, False, 1, False, label_saxs)
        
        cbar_ax2 = fig.add_axes([0.68, 0.05, 0.22, 0.03])
        cb2 = fig.colorbar(contour_waxs, cax=cbar_ax2, orientation='horizontal')
        setup_cbar(cb2, waxs_cmap_min, waxs_cmap_max, False, 2, False, label_waxs)

    if discharge_files is not None:
        for d_file in discharge_files:
            if starting_file <= d_file <= last_file:
                idx = d_file - starting_file
                d_y = y_data[idx]
                if elec_df_to_plot is not None and not elec_df_to_plot.empty:
                    axs[0].axhline(y=d_y, color='white', linestyle='--', linewidth=1.2, alpha=0.5)
                axs[1].axhline(y=d_y, color='white', linestyle='--', linewidth=1.2, alpha=0.5)
                axs[2].axhline(y=d_y, color='white', linestyle='--', linewidth=1.2, alpha=0.5)

    axs[0].tick_params(axis='both', labelsize=fntsize)
    axs[1].tick_params(axis='both', labelsize=fntsize)
    axs[2].tick_params(axis='both', labelsize=fntsize)
    
    if colorbar_position == 'bottom':
        fig.subplots_adjust(wspace=0.1, left=0.1, right=0.95, top=0.9, bottom=0.25)
    else:
        # Give more right padding for the outside colorbars
        fig.subplots_adjust(wspace=0.25, left=0.1, right=0.92, top=0.9, bottom=0.15)

    if save_plot:
        try:
            root = tk.Tk()
            root.withdraw()
            save_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=filename, filetypes=[("PNG files", "*.png")])
            if save_path:
                plt.savefig(save_path, dpi=600)
                print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Could not open save dialog, saving to current directory as {filename}")
            plt.savefig(filename, dpi=600)

    plt.show()

def q_to_two_theta(q_values, wavelength=0.15406):
    return 2 * np.degrees(np.arcsin((q_values * wavelength) / (4 * np.pi)))

def plot_3d_waterfall(
    data_dictionary, starting_file, last_file, discharge_file=None,
    q_min=16.9, q_max=19.8, offset_increment=0.000002,
    wavelength=0.15406, transparency=0.8,
    save_plot=False, filename="3D_Waterfall_Plot.png"
):
    """
    Generates a 3D waterfall plot for WAXS data.
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.title("3D Waterfall Plot of Filtered WAXS Data")

    base_offset = 0.0
    
    keys = list(range(starting_file, last_file + 1, 2))
    keys.reverse()
    
    if discharge_file is not None and discharge_file not in keys:
        keys.append(discharge_file)
        keys = sorted(keys, reverse=True)
        
    for ikey in keys:
        if ikey not in data_dictionary:
            continue
            
        original_q = 10 * data_dictionary[ikey].iloc[:, 0]
        original_Y = data_dictionary[ikey].iloc[:, 1]
        
        converted_two_theta = q_to_two_theta(original_q, wavelength)
        
        filtered_indices = (original_q >= q_min) & (original_q <= q_max)
        filtered_two_theta = converted_two_theta[filtered_indices]
        filtered_Y = original_Y[filtered_indices]
        
        offset = base_offset + (ikey - starting_file) * offset_increment
        Z = np.full_like(filtered_two_theta, offset)
        
        if discharge_file is not None and ikey == discharge_file:
            ax.plot(filtered_two_theta, Z, filtered_Y, color='black', linewidth=2.5, alpha=transparency)
        else:
            color = plt.cm.jet((ikey - starting_file) / max(1, last_file - starting_file))
            ax.plot(filtered_two_theta, Z, filtered_Y, color=color, alpha=transparency)

    ax.set_xlabel('2θ (degrees)', fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_zlabel('Intensity (a.u.)', fontsize=16)
    
    if discharge_file is not None:
        ax.plot([], [], [], color='black', linewidth=2.5, label=f'Discharge (File {discharge_file})')
        ax.legend(loc="upper right")

    ax.view_init(elev=18, azim=-106)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(filename, dpi=600)
        print(f"Plot saved as {filename}")
        
    plt.show()

def plot_2d_waterfall(
    data_dictionary, starting_file, discharge_file,
    q_min=16.96, q_max=21.12, intensity_min=0, intensity_max=0.02,
    offset_increment=0.00001, wavelength=0.15406,
    highlight_file=None, save_plot=False, filename="2D_Waterfall_Plot.png"
):
    """
    Generates a 2D stacked waterfall plot for WAXS data with fading colors.
    """
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if highlight_file is None:
        highlight_file = starting_file

    fig, ax = plt.subplots(figsize=(5,4))

    keys = list(range(starting_file, discharge_file + 1))
    
    if discharge_file not in keys:
        keys.append(discharge_file)
        keys = sorted(keys, reverse=False)

    num_curves = len(keys)
    cmap = plt.colormaps.get_cmap('Blues')
    norm_discharging = Normalize(vmin=starting_file, vmax=discharge_file + 1)
    
    for idx, ikey in enumerate(keys):
        if ikey not in data_dictionary:
            continue
            
        original_q = 10 * data_dictionary[ikey].iloc[:, 0]
        original_Y = data_dictionary[ikey].iloc[:, 1]

        converted_two_theta = q_to_two_theta(original_q, wavelength)

        valid_indices = (original_q >= q_min) & (original_q <= q_max) & (original_Y >= intensity_min) & (original_Y <= intensity_max)
        filtered_two_theta = converted_two_theta[valid_indices]
        filtered_Y = original_Y[valid_indices]

        if len(filtered_Y) > 7:
            smoothed_Y = savgol_filter(filtered_Y, window_length=20, polyorder=3) * 1000
        else:
            smoothed_Y = filtered_Y * 1000

        offset = idx * offset_increment

        if ikey == highlight_file:
            if len(filtered_Y) > 10:
                ax.plot(filtered_two_theta, savgol_filter(filtered_Y, window_length=10, polyorder=2) * 1000 + offset, color='black', linewidth=2.5, alpha=0.2)
            else:
                ax.plot(filtered_two_theta, filtered_Y * 1000 + offset, color='black', linewidth=2.5, alpha=0.2)
        else:
            color_idx = idx / max(1, num_curves)
            ax.plot(filtered_two_theta, smoothed_Y + offset, color=cmap(color_idx), alpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_discharging)
    sm.set_array([])

    axins = inset_axes(ax, width="3%", height="20%", loc='upper right',
                       bbox_to_anchor=(0, 0, 0.95, 0.95),
                       bbox_transform=ax.transAxes, borderpad=0)

    cbar = plt.colorbar(sm, cax=axins)
    cbar.set_ticks([])

    ax.set_xlim(24, 28)
    ax.set_xlabel('2θ (°)', fontsize=18)
    ax.set_ylabel('Intensity (a.u.)', fontsize=18)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='both', labelsize=18) 

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    plt.tight_layout()
    if save_plot:
        plt.savefig(filename, dpi=600)
        print(f"Plot saved as {filename}")

    plt.show()

def plot_selected_curves(data_dict, file_numbers, is_waxs=False, q_min=None, q_max=None, 
                         apply_smoothing=False, smoothing_method='savgol', 
                         window_length=15, polyorder=3, gaussian_sigma=2.0):
    """
    Plots specific curves by their file numbers from a given data dictionary.
    
    smoothing_method: 'savgol', 'gaussian', or 'moving_average'
    window_length: Used for savgol and moving_average. Must be odd.
    gaussian_sigma: Standard deviation for Gaussian kernel. Higher = more smoothing.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    
    plt.figure(figsize=(8, 6))
    wavelength = 0.15406  # Cu K-alpha in nm
    
    for f_num in file_numbers:
        if f_num not in data_dict:
            print(f"Warning: File {f_num} not found in the dictionary.")
            continue
            
        df = data_dict[f_num]
        x_data = df.iloc[:, 0] * 10  # Convert q to nm^-1
        y_data = df.iloc[:, 1]
        
        # Format the X-axis depending on if it is WAXS or SAXS
        if is_waxs:
            x_plot = 2 * np.degrees(np.arcsin((x_data * wavelength) / (4 * np.pi)))
            x_label = '2θ (°)'
        else:
            x_plot = x_data
            x_label = 'q (nm$^{-1}$)'
            
        # Optional cropping limits
        if q_min is not None and q_max is not None:
            valid_idx = (x_data >= q_min) & (x_data <= q_max)
            x_plot = x_plot[valid_idx]
            y_data = y_data[valid_idx]
            
        if apply_smoothing:
            if smoothing_method == 'savgol' and len(y_data) > window_length:
                y_plot = savgol_filter(y_data, window_length, polyorder)
            elif smoothing_method == 'gaussian':
                y_plot = gaussian_filter1d(y_data, sigma=gaussian_sigma)
            elif smoothing_method == 'moving_average':
                window = np.ones(window_length) / window_length
                y_plot = np.convolve(y_data, window, mode='same')
            else:
                y_plot = y_data
        else:
            y_plot = y_data
            
        plt.plot(x_plot, y_plot, label=f'File {f_num}')
        
    if not is_waxs:
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Selected SAXS Curves', fontsize=16)
    else:
        plt.title('Selected WAXS Curves', fontsize=16)
        
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Intensity (a.u.)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
