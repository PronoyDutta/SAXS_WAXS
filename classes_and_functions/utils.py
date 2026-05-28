import numpy as np
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
    elec_df_2=None, applied_current=None, active_material_mass=None
):
    """
    Generates a contour plot for SAXS/WAXS data correlated with electrochemical data.
    """
    if last_file is None:
        last_file = max(data_dictionary.keys())
        
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

    # Process data matrices
    for file_number in range(starting_file, last_file + 1):
        if data_normalization == 'relative' and Normalization_file is not None:
            norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1] / data_dictionary[Normalization_file].iloc[start_row:end_row + 1, 1]
        else:
            norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1]

        Intensity_list.append(norm_intensity.to_numpy())
        
        # Calculate time elapsed
        time_elapsed = float(keys[file_number - 1] - keys[starting_file - 1])
        time_t1.append(time_elapsed)

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
    if elec_df_2 is not None and not elec_df_2.empty:
        # Assume 'time/s' and 'Ewe/V' are columns in elec_df_2
        if 'time/s' in elec_df_2.columns and 'Ewe/V' in elec_df_2.columns:
            time_col = elec_df_2['time/s']
            volt_col = elec_df_2['Ewe/V']
        elif 6 in elec_df_2.columns and 2 in elec_df_2.columns: # HDF5 format fallback
            volt_col = elec_df_2[6]
            time_col = elec_df_2[2]
        else:
            time_col = elec_df_2.iloc[:, 0]
            volt_col = elec_df_2.iloc[:, 1]
            
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
