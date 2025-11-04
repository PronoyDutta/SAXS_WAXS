import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def find_nearest_time(target_time, sax_timestamps):
    """
    Find the SAXS file index with the closest timestamp to the target time.

    Parameters:
    target_time (float): Target time in seconds to match.
    sax_timestamps (list): List of SAXS file timestamps (datetime objects).

    Returns:
    int: Index of the closest timestamp.
    """
    # Convert sax_timestamps to seconds since the start
    sax_start_time = sax_timestamps[0]
    sax_timestamps_in_seconds = [(t - sax_start_time).total_seconds() for t in sax_timestamps]

    # Find the closest timestamp
    time_differences = [abs(target_time - t) for t in sax_timestamps_in_seconds]
    return time_differences.index(min(time_differences))



def process_electrochemical_data(all_data):
    """
    Process electrochemical data and link it with the nearest SAXS file timestamps.

    Parameters:
    all_data (dict): Dictionary containing electrochemical data and SAXS timestamps.

    Returns:
    dict: Processed electrochemical data linked with SAXS file numbers.
    """
    # Ask the user for the discharge rate
    rate = float(input("Enter discharge-charge rate, value of N in (C/N): "))

    electrochemical_df = all_data['Electrochemical'][0]
    sax_timestamps = all_data['Timestamps']

    # Filter the data to remove OCV periods (Ns != 0)
    elec_df = electrochemical_df[electrochemical_df['Ns'] != 0]

    # Detect cycle changes
    cycle_change_index = [elec_df.index[0]]
    absolute_times_for_electrochemical_state_change = [elec_df.loc[elec_df.index[0], 'time/s']]

    for line in range(len(elec_df['Ns']) - 1):
        if elec_df['Ns'].iloc[line] != elec_df['Ns'].iloc[line + 1]:
            cycle_change_index.append(elec_df.index[line + 1])
            absolute_times_for_electrochemical_state_change.append(
                elec_df.loc[elec_df.index[line + 1], 'time/s']
            )

    # Add the last index and time
    cycle_change_index.append(elec_df.index[-1])
    absolute_times_for_electrochemical_state_change.append(elec_df.loc[elec_df.index[-1], 'time/s'])

    # Convert absolute times to seconds relative to SAXS start time
    sax_start_time = sax_timestamps[0]
    absolute_times_in_seconds = [
        (datetime.fromtimestamp(t) - sax_start_time).total_seconds() if isinstance(t, (int, float)) else t
        for t in absolute_times_for_electrochemical_state_change
    ]

    # Link each cycle change with the nearest SAXS file
    saxs_file_numbers = [
        find_nearest_time(time, sax_timestamps)
        for time in absolute_times_for_electrochemical_state_change
    ]

    # Print details
    print("Indices for cycle change:", cycle_change_index)
    print("\nAbsolute times for change in electrochemical state:")
    print(absolute_times_for_electrochemical_state_change)
    print("\nCorresponding SAXS file numbers:")
    print(saxs_file_numbers)

    # Plot the electrochemical data
    Elec_Xdata = elec_df['time/s']
    Elec_Ydata = elec_df['Ewe/V']
    plt.plot(Elec_Xdata, Elec_Ydata)
    plt.xlabel('Time (s)')
    plt.ylabel('Potential (V)')
    plt.title('Electrochemical Data')
    plt.show()

    # Calculate applied current and active material mass
    applied_current = abs(elec_df['control/V/mA'].iloc[0])
    print("Applied current:", applied_current)

    active_material_mass = (applied_current * 1000 * rate) / 1675.0
    print(f"Mass of active material: {active_material_mass:.2f} mg")

    # Return processed data
    return {
        "filtered_dataframe": elec_df,
        "cycle_change_indices": cycle_change_index,
        "absolute_times_for_state_changes": absolute_times_for_electrochemical_state_change,
        "saxs_file_numbers": saxs_file_numbers,
        "applied_current": applied_current,
        "active_material_mass": active_material_mass,
    }
