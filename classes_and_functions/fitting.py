import numpy as np
import pandas as pd

def lorentzian(x, a, x0, gamma, baseline):
    return a * (0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma)**2) + baseline

def multi_lorentzian(x, *params):
    n_peaks = (len(params) - 1) // 3
    y = np.zeros_like(x)
    baseline = params[-1]
    for i in range(n_peaks):
        a = params[3*i]
        x0 = params[3*i + 1]
        gamma = params[3*i + 2]
        y += lorentzian(x, a, x0, gamma, 0)
    return y + baseline

def fit_waxs_peaks(data_dict, file_numbers, mean_positions, min_x, max_x, wavelength=0.15406, K=0.9, new_time_dict=None, applied_current=None, active_material_mass=None):
    from scipy.optimize import curve_fit
    
    df_curves = pd.DataFrame()
    df_scherrer = pd.DataFrame()
    
    valid_files = [f for f in file_numbers if f in data_dict]
    if not valid_files:
        print("No valid files to fit.")
        return df_curves, df_scherrer
    
    X_positions = data_dict[valid_files[0]].iloc[:, 0]
    indices_within_range = np.where((X_positions >= min_x) & (X_positions <= max_x))[0]
    
    if len(indices_within_range) == 0:
        print("No data found within the specified q-range.")
        return df_curves, df_scherrer
        
    start_row = np.min(indices_within_range)
    end_row = np.max(indices_within_range)
    
    amplitudes_for_avg = []
    
    for file_number in valid_files:
        x_filtered = data_dict[file_number].iloc[start_row:end_row+1, 0].values
        y_filtered = data_dict[file_number].iloc[start_row:end_row+1, 1].values
        
        # Initial guess
        initial_guess = []
        for pos in mean_positions:
            initial_guess += [0.001, pos, 0.1]
        initial_guess += [np.min(y_filtered)]

        # Bounds
        lower_bounds = []
        upper_bounds = []
        for pos in mean_positions:
            # We enforce the peak center to be within +/- 0.05 of the given mean_position
            lower_bounds += [0, pos - 0.05, 0.01]
            upper_bounds += [np.inf, pos + 0.05, 0.60]
        lower_bounds += [np.min(y_filtered)]
        upper_bounds += [np.max(y_filtered) + 0.1]

        bounds = (lower_bounds, upper_bounds)

        try:
            popt, pcov = curve_fit(
                multi_lorentzian, x_filtered, y_filtered,
                p0=initial_guess, bounds=bounds, max_nfev=100000
            )

            for i, pos in enumerate(mean_positions):
                amp = popt[3*i]
                center = popt[3*i + 1]
                gamma = popt[3*i + 2]
                baseline = popt[-1]
                perr = np.sqrt(np.diag(pcov))
                fwhm_err = perr[3*i + 2]

                df_curves.loc[file_number, f"Peak_{i+1}_Center"] = center
                df_curves.loc[file_number, f"Peak_{i+1}_FWHM"] = gamma
                df_curves.loc[file_number, f"Peak_{i+1}_Amplitude"] = amp
                df_curves.loc[file_number, f"Peak_{i+1}_Baseline"] = baseline
                df_curves.loc[file_number, f"Peak_{i+1}_FWHM error"] = fwhm_err

                amplitudes_for_avg.append(amp)

        except RuntimeError:
            print(f"Fit failed for file {file_number}")
            continue
        except Exception as e:
            print(f"Error during fit for file {file_number}: {e}")
            continue

    if amplitudes_for_avg:
        avg_amp = np.mean(amplitudes_for_avg)
        threshold = avg_amp / 10 # lower threshold so we don't skip plotting small peaks too aggressively
        
        file_to_time_dict = {}
        if new_time_dict is not None and len(valid_files) > 0:
            first_file = valid_files[0]
            if first_file in new_time_dict:
                from datetime import datetime
                try:
                    base_time = datetime.strptime(str(new_time_dict[first_file]), '%Y-%m-%dT%H:%M:%S')
                    for k, v in new_time_dict.items():
                        if k in valid_files:
                            current_time = datetime.strptime(str(v), '%Y-%m-%dT%H:%M:%S')
                            file_to_time_dict[k] = (current_time - base_time).total_seconds()
                except ValueError:
                    base_time = float(new_time_dict[first_file])
                    for k, v in new_time_dict.items():
                        if k in valid_files:
                            file_to_time_dict[k] = float(v) - base_time
            
        for file_number in valid_files:
            if file_number not in df_curves.index:
                continue
                
            for i in range(len(mean_positions)):
                try:
                    amp = df_curves.loc[file_number, f"Peak_{i+1}_Amplitude"]
                    fwhm = df_curves.loc[file_number, f"Peak_{i+1}_FWHM"]
                    center = df_curves.loc[file_number, f"Peak_{i+1}_Center"]

                    if amp >= threshold:
                        area = np.pi * amp * fwhm
                        df_curves.loc[file_number, f"Peak_{i+1}_Area"] = area

                        # Scherrer equation logic
                        theta_deg = 2 * np.degrees(np.arcsin((center * 10 * wavelength) / (4 * np.pi)))
                        beta_deg = 2 * np.degrees(np.arcsin(((center + fwhm/2) * 10 * wavelength) / (4 * np.pi))) - \
                                    2 * np.degrees(np.arcsin(((center - fwhm/2) * 10 * wavelength) / (4 * np.pi)))

                        beta_rad = np.radians(beta_deg)
                        theta_rad = np.radians(theta_deg / 2)
                        crystallite_size = (K * wavelength) / (beta_rad * np.cos(theta_rad))

                        df_scherrer.loc[file_number, f"Peak_{i+1}_Size"] = crystallite_size

                        # Error estimation
                        peak_fwhm_err = df_curves.loc[file_number, f"Peak_{i+1}_FWHM error"]
                        fwhm_plus = fwhm + peak_fwhm_err
                        fwhm_minus = fwhm - peak_fwhm_err
                        
                        beta_plus = 2 * np.degrees(np.arcsin(((center + fwhm_plus/2) * 10 * wavelength) / (4 * np.pi))) - \
                                    2 * np.degrees(np.arcsin(((center - fwhm_plus/2) * 10 * wavelength) / (4 * np.pi)))
                        beta_minus = 2 * np.degrees(np.arcsin(((center + fwhm_minus/2) * 10 * wavelength) / (4 * np.pi))) - \
                                        2 * np.degrees(np.arcsin(((center - fwhm_minus/2) * 10 * wavelength) / (4 * np.pi)))

                        beta_plus_rad = np.radians(beta_plus)
                        beta_minus_rad = np.radians(beta_minus)

                        crystallite_plus = (K * wavelength) / (beta_plus_rad * np.cos(theta_rad))
                        crystallite_minus = (K * wavelength) / (beta_minus_rad * np.cos(theta_rad))

                        crystallite_error = abs(crystallite_plus - crystallite_minus) / 2
                        df_scherrer.loc[file_number, f"Peak_{i+1}_Size_Error"] = crystallite_error

                except KeyError:
                    continue

            # Capacity calculation
            if new_time_dict is not None and applied_current is not None and active_material_mass is not None:
                time_in_s = file_to_time_dict.get(file_number, None)
                if time_in_s is not None:
                    capacity = (time_in_s * applied_current) / (active_material_mass * 3.6)
                    df_curves.loc[file_number, "Capacity (mAh/g)"] = capacity
                    df_scherrer.loc[file_number, "Capacity (mAh/g)"] = capacity
                    
    return df_curves, df_scherrer

def interactive_fit_viewer(data_dict, df_curves, file_numbers, mean_positions, min_x, max_x):
    from ipywidgets import interact, IntSlider
    import matplotlib.pyplot as plt
    import numpy as np

    valid_files = [f for f in file_numbers if f in data_dict and f in df_curves.index]
    if not valid_files:
        print("No valid fits found to plot.")
        return

    X_positions = data_dict[valid_files[0]].iloc[:, 0]
    indices = np.where((X_positions >= min_x) & (X_positions <= max_x))[0]
    start_row = np.min(indices)
    end_row = np.max(indices)

    def plot_overlay(file_key):
        if file_key not in valid_files:
            print(f"File {file_key} is not in the fitted dataset.")
            return

        x = data_dict[file_key].iloc[start_row:end_row+1, 0].values
        y = data_dict[file_key].iloc[start_row:end_row+1, 1].values

        plt.figure(figsize=(6, 4))
        plt.plot(x, y, label='Data', color='blue')

        try:
            baseline = df_curves.loc[file_key, f"Peak_1_Baseline"]
        except KeyError:
            baseline = 0
            
        y_fit_total = np.full_like(x, baseline)

        for i in range(len(mean_positions)):
            try:
                center = df_curves.loc[file_key, f"Peak_{i+1}_Center"]
                fwhm = df_curves.loc[file_key, f"Peak_{i+1}_FWHM"]
                amp = df_curves.loc[file_key, f"Peak_{i+1}_Amplitude"]
                
                # Try to get error for labeling, default to NaN if not present
                try:
                    fwhm_err = df_curves.loc[file_key, f"Peak_{i+1}_FWHM error"]
                except KeyError:
                    fwhm_err = float('nan')

                # Individual peak without baseline for the total sum, but plot with baseline for visualization
                y_peak = lorentzian(x, amp, center, fwhm, 0)
                y_fit_total += y_peak
                
                y_fit_plot = y_peak + baseline
                plt.plot(x, y_fit_plot, '--', label=f'Peak {i+1} Fit (q={center:.2f}, Err={fwhm_err:.3f})')
            except KeyError:
                continue

        plt.plot(x, y_fit_total, '-', label='Total Fit', color='red')
        plt.xlabel("q (nm⁻¹)")
        plt.ylabel("Intensity (a.u.)")
        plt.title(f"Overlayed Fit for File: {file_key}")
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.show()

    interact(plot_overlay, file_key=IntSlider(min=min(valid_files), max=max(valid_files), step=1, description='File Key', value=min(valid_files)))

def plot_peak_areas(df_curves, window_length=15, smoothing_method='savgol', polyorder=3, gaussian_sigma=3.0, filter_error_threshold=1.0, starting_cycle=None, final_cycle=None, ec_results=None, starting_file=None, last_file=None, plot_cycle_lines=True):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    
    df_plot = df_curves.copy()
    if ec_results is not None and starting_cycle is not None and final_cycle is not None:
        saxs_files = ec_results.get('saxs_file_numbers', [])
        if saxs_files and len(saxs_files) > 0:
            start_idx = max(0, int(2 * starting_cycle - 2))
            end_idx = min(len(saxs_files) - 1, int(2 * final_cycle))
            if start_idx < len(saxs_files) and end_idx < len(saxs_files):
                starting_file = saxs_files[start_idx]
                last_file = saxs_files[end_idx]
                
    if starting_file is not None:
        df_plot = df_plot[df_plot.index >= starting_file]
    if last_file is not None:
        df_plot = df_plot[df_plot.index <= last_file]
        
    plt.figure(figsize=(8, 5))
    
    is_size_plot = any('Size' in col for col in df_plot.columns)
    
    if is_size_plot:
        area_cols = [col for col in df_plot.columns if 'Size' in col and 'Peak' in col and 'Error' not in col]
        ylabel_text = 'Particle Size (Å)'
        title_text = 'Particle Size vs File Number'
    else:
        area_cols = [col for col in df_plot.columns if 'Area' in col and 'Peak' in col]
        ylabel_text = 'Area under Peak (a.u.)'
        title_text = 'Peak Area vs File Number'
    
    if not area_cols:
        print("No valid Peak Area or Particle Size columns found in the provided dataframe.")
        return
        
    for col in area_cols:
        # Create a working copy of the series
        y_series = df_plot[col].copy()
        
        # Filter out fits with high error if the threshold is provided
        if filter_error_threshold is not None:
            if is_size_plot:
                err_col = col + "_Error"
            else:
                err_col = col.replace('Area', 'FWHM error')
                
            if err_col in df_plot.columns:
                high_error_mask = df_plot[err_col] > filter_error_threshold
                y_series.loc[high_error_mask] = np.nan
                
        if window_length > 1:
            # Plot raw data as faint points
            plt.plot(df_plot.index, y_series, marker='o', linestyle='', alpha=0.3, color='tab:blue', label=f"{col.replace('_', ' ')} (Raw)")
            
            # Apply chosen smoothing method
            y_raw_for_smooth = y_series.interpolate(method='linear').fillna(0).values
            
            if smoothing_method == 'savgol':
                wl = window_length if window_length % 2 != 0 else window_length + 1
                smoothed = savgol_filter(y_raw_for_smooth, window_length=wl, polyorder=polyorder)
            elif smoothing_method == 'gaussian':
                smoothed = gaussian_filter1d(y_raw_for_smooth, sigma=gaussian_sigma)
            elif smoothing_method == 'moving_average':
                smoothed = y_series.rolling(window=window_length, center=True, min_periods=1).mean().values
            elif smoothing_method == 'moving_median':
                smoothed = y_series.rolling(window=window_length, center=True, min_periods=1).median().values
            else:
                smoothed = y_raw_for_smooth
                
            plt.plot(df_plot.index, smoothed, linestyle='-', linewidth=2, color='tab:red', label=f"{col.replace('_', ' ')} (Trend - {smoothing_method})")
        else:
            plt.plot(df_plot.index, y_series, marker='o', linestyle='-', label=col.replace('_', ' '))
            
    if ec_results is not None and plot_cycle_lines:
        saxs_files = ec_results.get('saxs_file_numbers', [])
        for i in range(1, len(saxs_files)):
            file_num = saxs_files[i]
            if not df_plot.empty and df_plot.index.min() <= file_num <= df_plot.index.max():
                if i % 2 != 0:
                    plt.axvline(x=file_num, color='green', linestyle=':', alpha=0.8, label='Discharge End' if i==1 else "")
                else:
                    plt.axvline(x=file_num, color='red', linestyle=':', alpha=0.8, label='Charge End' if i==2 else "")
                    
    # Handle duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    plt.xlabel('File Number', fontsize=14)
    plt.ylabel(ylabel_text, fontsize=14)
    plt.title(title_text, fontsize=16)
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def calculate_saxs_invariants(data_dict, file_numbers, q_range_intensity=(0.2, 4.0), q_range_mean=(0.2, 4.0), q_range_invariant=(0.2, 4.0), new_time_dict=None, applied_current=None, active_material_mass=None):
    from scipy.integrate import simpson
    import pandas as pd
    import numpy as np

    df_saxs = pd.DataFrame()
    valid_files = [f for f in file_numbers if f in data_dict]
    
    file_to_time_dict = {}
    if new_time_dict is not None and len(valid_files) > 0:
        first_file = valid_files[0]
        if first_file in new_time_dict:
            from datetime import datetime
            try:
                base_time = datetime.strptime(str(new_time_dict[first_file]), '%Y-%m-%dT%H:%M:%S')
                for k, v in new_time_dict.items():
                    if k in valid_files:
                        current_time = datetime.strptime(str(v), '%Y-%m-%dT%H:%M:%S')
                        file_to_time_dict[k] = (current_time - base_time).total_seconds()
            except ValueError:
                base_time = float(new_time_dict[first_file])
                for k, v in new_time_dict.items():
                    if k in valid_files:
                        file_to_time_dict[k] = float(v) - base_time

    for file_number in valid_files:
        # Assuming input q is in Å⁻¹, convert to nm⁻¹ as done in the old script
        q_nm = 10 * data_dict[file_number].iloc[:, 0].values
        I_vals = data_dict[file_number].iloc[:, 1].values
        
        # Intensity
        mask_int = (q_nm >= q_range_intensity[0]) & (q_nm <= q_range_intensity[1])
        if mask_int.sum() > 1:
            total_I = simpson(I_vals[mask_int], x=q_nm[mask_int])
        else:
            total_I = np.nan

        # Mean q
        mask_mean = (q_nm >= q_range_mean[0]) & (q_nm <= q_range_mean[1])
        if mask_mean.sum() > 1:
            denom = simpson(I_vals[mask_mean], x=q_nm[mask_mean])
            q_mean = simpson(q_nm[mask_mean] * I_vals[mask_mean], x=q_nm[mask_mean]) / denom if denom != 0 else np.nan
        else:
            q_mean = np.nan
            
        # Invariant
        mask_inv = (q_nm >= q_range_invariant[0]) & (q_nm <= q_range_invariant[1])
        if mask_inv.sum() > 1:
            invariant = simpson(q_nm[mask_inv]**2 * I_vals[mask_inv], x=q_nm[mask_inv])
        else:
            invariant = np.nan
        
        df_saxs.loc[file_number, "Integrated_Intensity"] = total_I
        df_saxs.loc[file_number, "Mean_q_nm-1"] = q_mean
        df_saxs.loc[file_number, "Invariant"] = invariant
        
        # Capacity calculation
        if new_time_dict is not None and applied_current is not None and active_material_mass is not None:
            time_in_s = file_to_time_dict.get(file_number, None)
            if time_in_s is not None:
                capacity = (time_in_s * applied_current) / (active_material_mass * 3.6)
                df_saxs.loc[file_number, "Capacity (mAh/g)"] = capacity
                
    return df_saxs

def plot_saxs_invariants(df_saxs, normalize=True, starting_cycle=None, final_cycle=None, ec_results=None, starting_file=None, last_file=None, plot_cycle_lines=True):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df_plot = df_saxs.copy()
    if ec_results is not None and starting_cycle is not None and final_cycle is not None:
        saxs_files = ec_results.get('saxs_file_numbers', [])
        if saxs_files and len(saxs_files) > 0:
            start_idx = max(0, int(2 * starting_cycle - 2))
            end_idx = min(len(saxs_files) - 1, int(2 * final_cycle))
            if start_idx < len(saxs_files) and end_idx < len(saxs_files):
                starting_file = saxs_files[start_idx]
                last_file = saxs_files[end_idx]
                
    if starting_file is not None:
        df_plot = df_plot[df_plot.index >= starting_file]
    if last_file is not None:
        df_plot = df_plot[df_plot.index <= last_file]
        
    if "Integrated_Intensity" in df_plot.columns:
        valid_data = df_plot["Integrated_Intensity"].dropna()
        if not valid_data.empty:
            if normalize:
                max_val = valid_data.max()
                min_val = valid_data.min()
                if max_val > min_val:
                    y_vals = ((df_plot["Integrated_Intensity"] - min_val) / (max_val - min_val)) * 100
                else:
                    y_vals = df_plot["Integrated_Intensity"]
                ylabel = 'Intensity (%)'
            else:
                y_vals = df_plot["Integrated_Intensity"]
                ylabel = 'Intensity (a.u.)'
            
            plt.figure(figsize=(6, 4))
            plt.plot(df_plot.index, y_vals, marker='o')
            if ec_results is not None and plot_cycle_lines:
                saxs_files = ec_results.get('saxs_file_numbers', [])
                for i in range(1, len(saxs_files)):
                    file_num = saxs_files[i]
                    if not df_plot.empty and df_plot.index.min() <= file_num <= df_plot.index.max():
                        if i % 2 != 0:
                            plt.axvline(x=file_num, color='green', linestyle=':', alpha=0.8)
                        else:
                            plt.axvline(x=file_num, color='red', linestyle=':', alpha=0.8)
            plt.title('Integrated Intensity')
            plt.xlabel('File Index')
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.show()

    if "Mean_q_nm-1" in df_plot.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(df_plot.index, df_plot["Mean_q_nm-1"], marker='o', color='orange')
        if ec_results is not None and plot_cycle_lines:
            saxs_files = ec_results.get('saxs_file_numbers', [])
            for i in range(1, len(saxs_files)):
                file_num = saxs_files[i]
                if not df_plot.empty and df_plot.index.min() <= file_num <= df_plot.index.max():
                    if i % 2 != 0:
                        plt.axvline(x=file_num, color='green', linestyle=':', alpha=0.8)
                    else:
                        plt.axvline(x=file_num, color='red', linestyle=':', alpha=0.8)
        plt.title('Mean q Position ⟨q⟩')
        plt.xlabel('File Index')
        plt.ylabel('q (nm⁻¹)')
        plt.grid(True)
        plt.show()

    if "Invariant" in df_plot.columns:
        valid_data = df_plot["Invariant"].dropna()
        if not valid_data.empty:
            if normalize:
                max_val = valid_data.max()
                min_val = valid_data.min()
                if max_val > min_val:
                    y_vals = ((df_plot["Invariant"] - min_val) / (max_val - min_val)) * 100
                else:
                    y_vals = df_plot["Invariant"]
                ylabel = 'Invariant (%)'
            else:
                y_vals = df_plot["Invariant"]
                ylabel = 'Invariant (a.u.)'
            
            plt.figure(figsize=(6, 4))
            plt.plot(df_plot.index, y_vals, marker='o', color='green')
            if ec_results is not None and plot_cycle_lines:
                saxs_files = ec_results.get('saxs_file_numbers', [])
                for i in range(1, len(saxs_files)):
                    file_num = saxs_files[i]
                    if not df_plot.empty and df_plot.index.min() <= file_num <= df_plot.index.max():
                        if i % 2 != 0:
                            plt.axvline(x=file_num, color='green', linestyle=':', alpha=0.8)
                        else:
                            plt.axvline(x=file_num, color='red', linestyle=':', alpha=0.8)
            plt.title('Scattering Invariant')
            plt.xlabel('File Index')
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.show()





def plot_combined_saxs_waxs_ec(df_saxs, df_curves, ec_results, all_data, starting_file=None, last_file=None, 
                               peak_col="Peak_1_Area", saxs_col="Integrated_Intensity",
                               apply_smoothing=True, window_length=25, filter_error_threshold=0.1, starting_cycle=None, final_cycle=None):
    if ec_results is not None and starting_cycle is not None and final_cycle is not None:
        saxs_files = ec_results.get('saxs_file_numbers', [])
        if saxs_files and len(saxs_files) > 0:
            start_idx = max(0, int(2 * starting_cycle - 2))
            end_idx = min(len(saxs_files) - 1, int(2 * final_cycle))
            if start_idx < len(saxs_files) and end_idx < len(saxs_files):
                starting_file = saxs_files[start_idx]
                last_file = saxs_files[end_idx]

    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    import pandas as pd

    if ec_results is None:
        print("EC results not provided.")
        return

    # Extract required constants
    applied_current = ec_results['applied_current']
    active_material_mass = ec_results['active_material_mass']

    # Filter EC dataframe to only include cycles between starting_file and last_file
    saxs_files = ec_results['saxs_file_numbers']
    cycle_indices = ec_results['cycle_change_indices']
    elec_df = ec_results['filtered_dataframe']
    
    start_idx = None
    end_idx = None
    
    for i, file in enumerate(saxs_files):
        if file >= starting_file and start_idx is None:
            start_idx = i
        if file >= last_file and end_idx is None:
            end_idx = i
            break
            
    if start_idx is None: start_idx = 0
    if end_idx is None or end_idx >= len(cycle_indices): end_idx = len(cycle_indices) - 1
        
    elec_df_filtered = elec_df.loc[cycle_indices[start_idx]:cycle_indices[end_idx]]

    time_col = 'time/s' if 'time/s' in elec_df_filtered.columns else elec_df_filtered.columns[2]
    volt_col = 'Ewe/V' if 'Ewe/V' in elec_df_filtered.columns else elec_df_filtered.columns[6]

    # Convert EC time to capacity
    ec_time = elec_df_filtered[time_col].values
    ec_time_relative = ec_time - ec_time[0]
    ec_capacity = (ec_time_relative * applied_current) / (active_material_mass * 3.6)
    ec_voltage = elec_df_filtered[volt_col].values

    # Recalculate true capacities for the x-axis to guarantee alignment with EC
    time_dict = all_data.get('TimeStamps_SAXS')
    if time_dict is None:
        print("Could not find TimeStamps_SAXS to align capacities.")
        return
        
    try:
        base_time = datetime.strptime(str(time_dict[starting_file]), '%Y-%m-%dT%H:%M:%S')
        is_string = True
    except ValueError:
        base_time = float(time_dict[starting_file])
        is_string = False

    df_combined = df_curves.copy()
    for col in df_saxs.columns:
        if col not in df_combined.columns:
            df_combined[col] = df_saxs[col]
        else:
            df_combined[col] = df_combined[col].combine_first(df_saxs[col])

    # Truncate to just the file range
    try:
        df_combined = df_combined.loc[starting_file:last_file]
    except KeyError:
        pass # Handle cases where index doesn't exactly match

    cap_vals = []
    valid_indices = []
    for idx in df_combined.index:
        if idx >= starting_file and idx <= last_file:
            if idx in time_dict:
                if is_string:
                    current_time = datetime.strptime(str(time_dict[idx]), '%Y-%m-%dT%H:%M:%S')
                    t_rel = (current_time - base_time).total_seconds()
                else:
                    t_rel = float(time_dict[idx]) - base_time
                
                cap = (t_rel * applied_current) / (active_material_mass * 3.6)
                cap_vals.append(cap)
                valid_indices.append(idx)
            
    df_plot = df_combined.loc[valid_indices].copy()
    df_plot['True_Capacity'] = cap_vals

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot EC curve
    ax1.plot(ec_capacity, ec_voltage, color='grey', linestyle='-.', label='Potential (V vs. Li/Li+)')
    ax1.set_xlabel('Specific capacity (mAh g$^{-1}$)', fontsize=14)
    ax1.set_ylabel('Potential (V vs. Li/Li+)', fontsize=14)
    ax1.set_ylim(np.min(ec_voltage) - 0.2, np.max(ec_voltage) + 0.2)
    ax1.set_xlim(-0.05 * np.max(ec_capacity), 1.05 * np.max(ec_capacity))

    # Secondary y-axis for percentages
    ax2 = ax1.twinx()
    ax2.set_ylabel('Intensity & Area (%)', fontsize=14)
    ax2.set_ylim(0, 100)

    # Plot SAXS
    if saxs_col in df_plot.columns and not df_plot[saxs_col].dropna().empty:
        saxs_data = df_plot[saxs_col].copy()
        if apply_smoothing:
            saxs_data = saxs_data.rolling(window=window_length, center=True, min_periods=1).median()
            
        valid_idx = ~saxs_data.isna()
        saxs_max = saxs_data[valid_idx].max()
        saxs_min = saxs_data[valid_idx].min()
        if saxs_max != saxs_min:
            saxs_pct = ((saxs_data[valid_idx] - saxs_min) / (saxs_max - saxs_min)) * 100
        else:
            saxs_pct = saxs_data[valid_idx] * 0
        ax2.plot(df_plot['True_Capacity'][valid_idx], saxs_pct, color='tab:red', marker='o', linestyle='-', label=f'SAXS {saxs_col} (%)')

    # Plot WAXS
    if peak_col in df_plot.columns and not df_plot[peak_col].dropna().empty:
        waxs_data = df_plot[peak_col].copy()
        
        # Filter high error
        if filter_error_threshold is not None:
            err_col = peak_col.replace('Area', 'FWHM error')
            if err_col in df_plot.columns:
                high_error_mask = df_plot[err_col] > filter_error_threshold
                waxs_data.loc[high_error_mask] = np.nan
                
        if apply_smoothing:
            waxs_data = waxs_data.rolling(window=window_length, center=True, min_periods=1).median()
            
        valid_idx = ~waxs_data.isna()
        if waxs_data[valid_idx].empty:
            print("No valid WAXS data left after filtering.")
        else:
            waxs_max = waxs_data[valid_idx].max()
            waxs_min = waxs_data[valid_idx].min()
            if waxs_max != waxs_min:
                waxs_pct = ((waxs_data[valid_idx] - waxs_min) / (waxs_max - waxs_min)) * 100
            else:
                waxs_pct = waxs_data[valid_idx] * 0
            ax2.plot(df_plot['True_Capacity'][valid_idx], waxs_pct, color='tab:blue', marker='o', linestyle='-', label=f'WAXS {peak_col} (%)')

    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=12)

    plt.title('Combined SAXS / WAXS / Electrochemistry', fontsize=16, pad=30)
    plt.tight_layout()
    plt.show()
