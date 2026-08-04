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
        if new_time_dict is not None:
            file_to_time_dict = {v: k for k, v in new_time_dict.items()}
            
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
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.show()

    interact(plot_overlay, file_key=IntSlider(min=min(valid_files), max=max(valid_files), step=1, description='File Key', value=min(valid_files)))

def plot_peak_areas(df_curves, window_length=15, smoothing_method='savgol', polyorder=3, gaussian_sigma=3.0, filter_error_threshold=1.0):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    
    plt.figure(figsize=(8, 5))
    
    # Find all columns that match 'Peak_X_Area'
    area_cols = [col for col in df_curves.columns if 'Area' in col and 'Peak' in col]
    
    if not area_cols:
        print("No peak areas found in the dataframe. The fits might have failed or peaks were below threshold.")
        return
        
    for col in area_cols:
        # Create a working copy of the series
        y_series = df_curves[col].copy()
        
        # Filter out fits with high error if the threshold is provided
        if filter_error_threshold is not None:
            err_col = col.replace('Area', 'FWHM error')
            if err_col in df_curves.columns:
                high_error_mask = df_curves[err_col] > filter_error_threshold
                y_series.loc[high_error_mask] = np.nan
                
        if window_length > 1:
            # Plot raw data as faint points
            plt.plot(df_curves.index, y_series, marker='o', linestyle='', alpha=0.3, color='tab:blue', label=f"{col.replace('_', ' ')} (Raw)")
            
            # Apply chosen smoothing method
            # For rolling window methods, keeping NaNs is usually fine as long as min_periods=1,
            # but for savgol/gaussian, NaNs will cause issues. We interpolate or fill for the smoothing layer.
            y_raw_for_smooth = y_series.interpolate(method='linear').fillna(0).values
            
            if smoothing_method == 'savgol':
                # savgol requires window_length to be odd
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
                
            plt.plot(df_curves.index, smoothed, linestyle='-', linewidth=2, color='tab:red', label=f"{col.replace('_', ' ')} (Trend - {smoothing_method})")
        else:
            plt.plot(df_curves.index, y_series, marker='o', linestyle='-', label=col.replace('_', ' '))
        
    plt.xlabel('File Number', fontsize=14)
    plt.ylabel('Area under Peak (a.u.)', fontsize=14)
    plt.title('Peak Area vs File Number', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
