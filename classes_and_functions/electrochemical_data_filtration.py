import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from .utils import find_nearest_time

def process_electrochemical_data(all_data, rate):
    """
    Process electrochemical data and link it with the nearest SAXS file timestamps.

    Parameters:
    all_data (dict): Dictionary containing electrochemical data and SAXS timestamps.
    rate (float): Discharge-charge rate (value of N in C/N).

    Returns:
    dict: Processed electrochemical data linked with SAXS file numbers.
    """
    electrochemical_df = all_data.get('Electrochemical')
    if electrochemical_df is None or electrochemical_df.empty:
        print("No electrochemical data found.")
        return None

    sax_timestamps = all_data.get('TimeStamps_SAXS', {})

    # Check for the correct column names, which differ between individual MPR and HDF5 extraction
    if 'Ns' in electrochemical_df.columns:
        ns_col = 'Ns'
        time_col = 'time/s'
        volt_col = 'Ewe/V'
        ctrl_col = 'control/V/mA'
    elif 9 in electrochemical_df.columns: # fallback for HDF5 index-based columns if headers are missing
        ns_col = 9
        time_col = 2 # typically 'time/s'
        volt_col = 6 # typically 'Ewe/V'
        ctrl_col = 5 # typically 'I/mA'
    else:
        # Just use positional as a wild guess or return early
        print("Warning: Could not identify standard electrochemical columns.")
        return None

    # Filter the data to remove OCV periods (Ns != 0)
    elec_df = electrochemical_df[electrochemical_df[ns_col] != 0]
    if elec_df.empty:
        print("No non-OCV electrochemical data found.")
        return None

    # Detect cycle changes
    cycle_change_index = [elec_df.index[0]]
    absolute_times_for_electrochemical_state_change = [elec_df.loc[elec_df.index[0], time_col]]

    for line in range(len(elec_df[ns_col]) - 1):
        if elec_df[ns_col].iloc[line] != elec_df[ns_col].iloc[line + 1]:
            cycle_change_index.append(elec_df.index[line + 1])
            absolute_times_for_electrochemical_state_change.append(
                elec_df.loc[elec_df.index[line + 1], time_col]
            )

    # Add the last index and time
    cycle_change_index.append(elec_df.index[-1])
    absolute_times_for_electrochemical_state_change.append(elec_df.loc[elec_df.index[-1], time_col])

    # Convert absolute times to seconds relative to SAXS start time
    keys = list(sax_timestamps.keys())
    if not keys:
        print("No SAXS timestamps found.")
        return None

    # Try to calculate start time from timestamp format
    try:
        sax_start_time = datetime.strptime(str(sax_timestamps[keys[0]]), '%Y-%m-%dT%H:%M:%S')
        absolute_times_in_seconds = [
            (datetime.fromtimestamp(t) - sax_start_time).total_seconds() if isinstance(t, (int, float)) else t
            for t in absolute_times_for_electrochemical_state_change
        ]
    except ValueError:
        # Fallback if timestamps are already numeric or different format
        sax_start_time = float(sax_timestamps[keys[0]])
        absolute_times_in_seconds = absolute_times_for_electrochemical_state_change

    # Link each cycle change with the nearest SAXS file
    saxs_file_numbers = [
        find_nearest_time(time, sax_timestamps)[0]
        for time in absolute_times_for_electrochemical_state_change
    ]

    # Print details
    print("Indices for cycle change:", cycle_change_index)
    print("\nAbsolute times for change in electrochemical state:")
    print(absolute_times_for_electrochemical_state_change)
    print("Corresponding SAXS file numbers:")
    print(saxs_file_numbers)

    # Identify discharge files by checking if voltage dropped during the state
    discharge_saxs_files = []
    for i in range(len(cycle_change_index) - 1):
        start_idx = cycle_change_index[i]
        end_idx = cycle_change_index[i+1]
        start_v = elec_df.loc[start_idx, volt_col]
        end_v = elec_df.loc[end_idx, volt_col]
        if end_v < start_v:
            discharge_saxs_files.append(saxs_file_numbers[i+1])
    
    print("\nIdentified discharge end SAXS files:")
    print(discharge_saxs_files)

    # Plot the electrochemical data
    Elec_Xdata = elec_df[time_col]
    Elec_Ydata = elec_df[volt_col]
    plt.plot(Elec_Xdata, Elec_Ydata)
    plt.xlabel('Time (s)')
    plt.ylabel('Potential (V)')
    plt.title('Electrochemical Data')
    plt.show()

    # Calculate applied current and active material mass
    applied_current = abs(elec_df[ctrl_col].iloc[0])
    print("Applied current:", applied_current)

    active_material_mass = (applied_current * 1000 * rate) / 1675.0
    print(f"Mass of active material: {active_material_mass:.2f} mg")

    # Return processed data
    return {
        "filtered_dataframe": elec_df,
        "cycle_change_indices": cycle_change_index,
        "absolute_times_for_state_changes": absolute_times_for_electrochemical_state_change,
        "saxs_file_numbers": saxs_file_numbers,
        "discharge_saxs_files": discharge_saxs_files,
        "applied_current": applied_current,
        "active_material_mass": active_material_mass,
    }
