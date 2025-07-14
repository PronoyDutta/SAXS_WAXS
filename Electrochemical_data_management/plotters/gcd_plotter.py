import os
import pandas as pd

def load_gcd_data(project_dir, sample_names, cycle_numbers=None):
    """
    Loads GCD data for multiple samples.
    Returns a dict of {sample_name: dataframe} and {sample_name: mass}
    """
    data = {}
    masses = {}

    for sample in sample_names:
        gcd_path = os.path.join(project_dir, sample, "GCD", "data.csv")
        meta_path = os.path.join(project_dir, sample, "GCD", "meta.json")

        if os.path.isfile(gcd_path):
            df = pd.read_csv(gcd_path)
            if cycle_numbers:
                df = df[df['cycle_number'].isin(cycle_numbers)]

            data[sample] = df

            if os.path.isfile(meta_path):
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                masses[sample] = meta.get("mass_mg", 1.0)  # fallback to 1.0 mg
            else:
                masses[sample] = 1.0  # fallback

    return data, masses

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_gcd_curves(groups):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Build color list per group
    color_map = {}
    total_cycles = sum(
        df['cycle_number'].nunique() - (1 if 0 in df['cycle_number'].unique() else 0)
        for g in groups for df in g['data'].values()
    )
    cmap = cm.get_cmap("nipy_spectral", total_cycles)
    
    color_keys = []
    for group in groups:
        for sample, df in group["data"].items():
            for cyc in df['cycle_number'].unique():
                if cyc != 0:
                    color_keys.append(f"{sample}_C{cyc}_{group['label']}")
    
    color_keys = sorted(set(color_keys))
    for i, key in enumerate(color_keys):
        color_map[key] = cmap(i)

    # Plot all groups
    for group in groups:
        group_label = group["label"]
        for sample, df in group["data"].items():
            mass_g = float(group["masses"].get(sample, 1.0)) / 1000
            df = df.copy()
            df['specific_capacity'] = (df['capacity_mAh'] / mass_g).abs()

            grouped = df.groupby(['cycle_number', 'half_cycle'])

            for (cyc, half), subdf in grouped:
                if cyc == 0 or half == 0:
                    continue
                color_key = f"{sample}_C{cyc}_{group_label}"
                color = color_map[color_key]
                label = f"{group_label} - Cycle {cyc}" if half == 1 else None
                ax.plot(subdf['specific_capacity'], subdf['voltage_V'], color=color, label=label)

    ax.set_xlabel("Specific Capacity (mAh/g)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("GCD Curves")
    ax.legend()
    ax.grid(True)
    return fig, ax


