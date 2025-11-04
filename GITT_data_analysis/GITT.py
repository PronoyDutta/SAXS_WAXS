import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Function to select file using Tkinter
def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select GITT Data File",
                                           filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")])
    return file_path

# Load GITT Data
file_path = select_file()
if not file_path:
    print("No file selected. Exiting...")
    exit()

# Determine file type and read the data
if file_path.endswith(".xlsx"):
    df = pd.read_excel(file_path)
elif file_path.endswith(".csv"):
    df = pd.read_csv(file_path)
else:
    print("Unsupported file format. Please select an Excel or CSV file.")
    exit()

# Convert time from hours to seconds
df["time/s"] = df["time/h"] * 3600

# Compute dE/dt to detect pulses
df["dE/dt"] = np.gradient(df["Ewe/V"], df["time/s"])

# Adaptive threshold to detect pulses
threshold = df["dE/dt"].std() * 3  # Adjust if needed
pulse_indices = df.index[np.abs(df["dE/dt"]) > threshold].tolist()

# Split pulses and relaxation phases
pulse_segments, relaxation_segments = [], []

for i in range(len(pulse_indices) - 1):
    start_idx, end_idx = pulse_indices[i], pulse_indices[i + 1]

    # Pulse segment
    pulse_segments.append(df.iloc[start_idx:end_idx])

    # Relaxation segment (immediately after pulse)
    relaxation_segments.append(df.iloc[end_idx:end_idx + (end_idx - start_idx)])

# Define assumed diffusion length L (in cm)
L_nm = 100  # Typical Li-S electrode particle size in nm
L_cm = L_nm * 1e-7  # Convert nm to cm

# Compute D_Li for each pulse (with IR drop correction)
D_Li_values, capacity_values = [], []

# Constant current value (provided earlier)
current_A = 87.3e-6  # 87.3 microamps

for pulse, relax in zip(pulse_segments, relaxation_segments):
    if relax.empty or pulse.empty:
        continue

    tau = relax["time/s"].iloc[-1] - relax["time/s"].iloc[0]  # Relaxation time (s)
    
    # IR drop correction: Take last voltage in pulse & first voltage in relaxation
    E_pulse_end = pulse["Ewe/V"].iloc[-1]
    E_relax_start = relax["Ewe/V"].iloc[0]

    delta_E_corrected = E_pulse_end - E_relax_start  # IR-corrected voltage step
    delta_E_relax = relax["Ewe/V"].iloc[-1] - relax["Ewe/V"].iloc[0]  # Relaxation voltage change

    # Compute capacity (Q = I * t)
    pulse_time = pulse["time/s"].iloc[-1] - pulse["time/s"].iloc[0]
    capacity = current_A * pulse_time  # Coulombs

    # Calculate D_Li using corrected voltage step
    if tau > 0 and delta_E_relax != 0:
        D_Li = (4 / np.pi) * ((L_cm**2) / tau) * (delta_E_corrected / delta_E_relax) ** 2
        D_Li_values.append(D_Li)
        capacity_values.append(capacity)

# Store results in a DataFrame
results_df = pd.DataFrame({
    "Pulse Index": range(len(D_Li_values)),
    "D_Li (cm²/s)": D_Li_values,
    "Capacity (C)": capacity_values
})

# Save results to an Excel file
output_file = "Li_diffusion_results_corrected.xlsx"
results_df.to_excel(output_file, index=False)

# Plot D_Li vs. Capacity
plt.figure(figsize=(8, 6))
plt.scatter(results_df["Capacity (C)"], results_df["D_Li (cm²/s)"], marker='o', label="Data Points")
plt.plot(results_df["Capacity (C)"], results_df["D_Li (cm²/s)"], linestyle='-', alpha=0.7)

plt.xlabel("Capacity (C)")
plt.ylabel("Diffusion Coefficient \(D_{Li}\) (cm²/s)")
plt.title("Lithium Diffusion Coefficient vs. Capacity (IR Corrected)")
plt.grid(True)
plt.legend()
plt.show()

# Print summary
print(f"IR-Corrected Li Diffusion Coefficients saved to '{output_file}'")
