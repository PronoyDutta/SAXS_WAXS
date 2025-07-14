import matplotlib.pyplot as plt
import numpy as np

def plot_rate_performance(all_groups, theoretical_capacity, cycle_start, cycle_end):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.nipy_spectral
    colors = [cmap(i / len(all_groups)) for i in range(len(all_groups))]

    for idx, group in enumerate(all_groups):
        group_label = group["label"]
        data_dict = group["data"]
        mass_dict = group["masses"]
        color = colors[idx]

        x_vals = []
        y_vals = []
        annotations = []

        cycle_offset = 0

        for label, df in data_dict.items():
            df = df.copy()
            df = df[(df["cycle_number"] >= cycle_start) & (df["cycle_number"] <= cycle_end)]
            df = df[df["half_cycle"] == 1]  # only discharge

            if df.empty:
                continue

            mass_g = float(mass_dict.get(label, 1.0)) / 1000  # mg to g

            try:
                current = abs(df["current_mA"].iloc[0])
            except:
                current = 1.0

            c_rate = current / (theoretical_capacity * mass_g)

            grouped = df.groupby("cycle_number")
            group_x = []
            group_y = []

            for cyc, g in grouped:
                cap_mAh = g["capacity_mAh"].iloc[-1]
                spec_cap = cap_mAh / mass_g
                group_x.append(cycle_offset + cyc)
                group_y.append(spec_cap)

            if group_x:
                mid_x = np.mean(group_x)
                mid_y = np.max(group_y)
                annotations.append((mid_x, mid_y + 10, f"{c_rate:.2f}C"))

            x_vals.extend(group_x)
            y_vals.extend(group_y)

            if group_x:
                cycle_offset = x_vals[-1]

        ax.plot(x_vals, y_vals, 'o-', label=group_label, color=color)
        for x, y, label in annotations:
            ax.text(x, y, label, ha="center", fontsize=9, color=color)

    ax.set_xlabel("Cycle Number")
    ax.set_ylabel("Specific Capacity (mAh/g)")
    ax.set_title("Rate Performance")
    ax.grid(True)
    ax.legend()
    return fig, ax
