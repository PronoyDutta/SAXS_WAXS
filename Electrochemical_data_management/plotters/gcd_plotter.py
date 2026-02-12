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
from matplotlib.font_manager import FontProperties  # for +1 pt font adjustments

def plot_gcd_curves(
    groups,
    grid: bool = True,
    color_scheme: str = "nipy_spectral",
    custom_colors=None,
    legend_loc: str = "upper right",
    legend_xy: tuple | None = None,
    legend_draggable: bool = False,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
):
    # Match the rate plot figure sizing for visual consistency
    fig, ax = plt.subplots(figsize=(4, 3))

    # Build color list per group
    total_cycles = sum(
        df['cycle_number'].nunique() - (1 if 0 in df['cycle_number'].unique() else 0)
        for g in groups for df in g['data'].values()
    )
    cmap = cm.get_cmap(color_scheme, total_cycles) if color_scheme != "default" else None

    color_keys = []
    for group in groups:
        for sample, df in group["data"].items():
            for cyc in df['cycle_number'].unique():
                if cyc != 0:
                    color_keys.append(f"{sample}_C{cyc}_{group['label']}")
    color_keys = sorted(set(color_keys))

    color_map = {}
    # If default scheme, use Matplotlib's default color cycle explicitly
    default_cycle_colors = None
    if color_scheme == "default":
        try:
            prop_cycle = plt.rcParams.get('axes.prop_cycle', None)
            if prop_cycle is not None:
                default_cycle_colors = prop_cycle.by_key().get('color', None)
            else:
                default_cycle_colors = None
        except Exception:
            default_cycle_colors = None

    for i, key in enumerate(color_keys):
        if custom_colors is not None and key in custom_colors:
            color_map[key] = custom_colors[key]
        elif cmap is not None:
            color_map[key] = cmap(i)
        elif default_cycle_colors:
            color_map[key] = default_cycle_colors[i % len(default_cycle_colors)]
        else:
            color_map[key] = None  # fallback to matplotlib default

    # Plot all groups
    for group in groups:
        group_label = group["label"]
        for sample, df in group["data"].items():
            mass_g = float(group["masses"].get(sample, 1.0)) / 1000
            df = df.copy()
            df['specific_capacity'] = (df['capacity_mAh'] / mass_g).abs()
            # Only consider half_cycle 1 (discharge) and 2 (charge)
            df = df[df['half_cycle'].isin([1, 2])]
            grouped = df.groupby(['cycle_number', 'half_cycle'])

            for (cyc, half), subdf in grouped:
                if cyc == 0:
                    continue
                color_key = f"{sample}_C{cyc}_{group_label}"
                color = color_map.get(color_key, None)
                label = f"{group_label}" if half == 1 else None
                #label = f"{group_label} - Cycle {cyc}" if half == 1 else None
                # Always pass an explicit color if we resolved one so charge/discharge match
                if color is not None:
                    ax.plot(subdf['specific_capacity'], subdf['voltage_V'], color=color, label=label)
                else:
                    ax.plot(subdf['specific_capacity'], subdf['voltage_V'], label=label)

    ax.set_xlabel("Specific Capacity (mAh g$^{-1}$)")
    ax.set_ylabel("Potential (V vs. Li/Li+)")
    ax.set_title("GCD Curves")
    # Increase axis label and tick label fontsize by +1 pt (mirrors rate plot behavior)
    try:
        label_size_base = FontProperties(size=plt.rcParams.get('axes.labelsize', plt.rcParams.get('font.size', 10.0))).get_size_in_points()
        tick_size_base = FontProperties(size=plt.rcParams.get('xtick.labelsize', plt.rcParams.get('font.size', 10.0))).get_size_in_points()
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_size_base + 1)
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_size_base + 1)
        ax.tick_params(axis='both', labelsize=tick_size_base + 1)
    except Exception:
        ax.tick_params(axis='both', labelsize=11)
    # Apply user-defined axis ranges if provided
    try:
        if xlim is not None and len(xlim) == 2:
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        if ylim is not None and len(ylim) == 2:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
    except Exception:
        pass
    # Legend: no frame. If manual position provided, anchor to (x,y) in axes coords
    if legend_xy is not None:
        leg = ax.legend(frameon=False, loc="center", bbox_to_anchor=legend_xy, bbox_transform=ax.transAxes)
    else:
        leg = ax.legend(frameon=False, loc=legend_loc)
    if legend_draggable and leg is not None:
        try:
            leg.set_draggable(True)
        except Exception:
            pass
    ax.grid(grid)
    return fig, ax


