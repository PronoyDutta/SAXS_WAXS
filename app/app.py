import os
import json
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Attempt to import existing plotting utilities
try:
    from Electrochemical_data_management.plotters import gcd_plotter, rate_plotter  # type: ignore
except Exception:
    gcd_plotter = None  # type: ignore
    rate_plotter = None  # type: ignore

STATE_FILE = os.path.expanduser("~/.saxs_waxs_gui_state.json")

###############################################################################
# Helper functions
###############################################################################

def load_previous_project() -> str | None:
    try:
        if os.path.isfile(STATE_FILE):
            with open(STATE_FILE) as f:
                cfg = json.load(f)
            p = cfg.get("last_project_folder")
            if p and os.path.isdir(p):
                return p
    except Exception:
        pass
    return None

def save_previous_project(path: str) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"last_project_folder": path}, f)
    except Exception:
        pass

def list_samples(base: str, plot_type: str) -> List[str]:
    out: List[str] = []
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        sdir = os.path.join(base, name)
        if not os.path.isdir(sdir):
            continue
        if plot_type == "GCD":
            if any(os.path.isdir(os.path.join(sdir, sub)) for sub in ["GCD", "Rate"]):
                out.append(name)
        else:
            if os.path.isdir(os.path.join(sdir, plot_type)):
                out.append(name)
    return out

def list_csv_files(base: str, sample: str, plot_type: str) -> List[str]:
    sample_dir = os.path.join(base, sample)
    files: List[str] = []
    if not os.path.isdir(sample_dir):
        return files
    if plot_type == "GCD":
        for sub in ["GCD", "Rate"]:
            subdir = os.path.join(sample_dir, sub)
            if os.path.isdir(subdir):
                for f in os.listdir(subdir):
                    if f.endswith('.csv'):
                        files.append(os.path.join(sub, f))
    else:
        subdir = os.path.join(sample_dir, plot_type)
        if os.path.isdir(subdir):
            for f in os.listdir(subdir):
                if f.endswith('.csv'):
                    files.append(os.path.join(plot_type, f))
    return sorted(files)

def load_group_data(base: str, sample: str, rel_paths: List[str], cycles: List[int] | None) -> tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    data: Dict[str, pd.DataFrame] = {}
    masses: Dict[str, float] = {}
    for rel in rel_paths:
        full_path = os.path.join(base, sample, rel)
        folder = os.path.dirname(full_path)
        meta_pkl = os.path.join(folder, 'metadata.pkl')
        meta_json = os.path.join(folder, 'meta.json')
        mass_mg = 1.0
        try:
            if os.path.isfile(meta_pkl):
                import pickle
                with open(meta_pkl, 'rb') as f:
                    meta = pickle.load(f)
                mass_mg = float(meta.get('mass_mg', 1.0))
            elif os.path.isfile(meta_json):
                with open(meta_json) as f:
                    meta = json.load(f)
                mass_mg = float(meta.get('mass_mg', 1.0))
        except Exception:
            pass
        if not os.path.isfile(full_path):
            st.warning(f"Missing file: {rel}")
            continue
        try:
            # First attempt: default pandas parser
            df = pd.read_csv(full_path)
        except Exception as e1:
            # Fallback: more permissive parser with automatic delimiter inference
            try:
                df = pd.read_csv(full_path, sep=None, engine='python', encoding_errors='ignore')
            except Exception as e2:
                st.error(f"Failed to load {rel}: {type(e2).__name__}: {e2}")
                continue

        try:
            if cycles:
                if 'cycle_number' in df.columns:
                    df = df[df['cycle_number'].astype(str).isin([str(c) for c in cycles])]
            key = f"{sample}_{os.path.splitext(os.path.basename(rel))[0]}"
            data[key] = df
            masses[key] = mass_mg
        except Exception as e:
            st.error(f"Failed to prepare data for {rel}: {type(e).__name__}: {e}")
    return data, masses

def ensure_fig_dir(base: str) -> str:
    fig_dir = os.path.join(base, 'Figures') if os.path.isdir(base) else os.path.join(os.getcwd(), 'Figures')
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir

def unique_path(directory: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    out = os.path.join(directory, filename)
    i = 1
    while os.path.exists(out):
        out = os.path.join(directory, f"{base}_{i}{ext}")
        i += 1
    return out

###############################################################################
# Streamlit App
###############################################################################

def main():
    st.set_page_config(page_title="Electrochemical Plotter", layout="wide")
    st.title("Electrochemical Plotter")

    st.sidebar.header("Configuration")
    default_project = load_previous_project() or ""
    project_folder = st.sidebar.text_input("Project Folder", value=default_project, help="Base directory containing sample subfolders")
    if st.sidebar.button("Set Folder"):
        if os.path.isdir(project_folder):
            save_previous_project(project_folder)
            st.sidebar.success("Folder saved.")
        else:
            st.sidebar.error("Not a valid directory.")

    plot_type = st.sidebar.selectbox("Plot Type", ["GCD", "Rate", "CV"], index=0)

    samples = list_samples(project_folder, plot_type)
    sample = st.sidebar.selectbox("Sample", samples) if samples else None
    files: List[str] = []
    if sample:
        files = list_csv_files(project_folder, sample, plot_type)
    selected_files = st.sidebar.multiselect("File(s)", files)
    cycles_text = st.sidebar.text_input("Cycles (comma separated)", value="")
    try:
        cycles = [int(c.strip()) for c in cycles_text.split(',') if c.strip()]
    except Exception:
        cycles = None
    label = st.sidebar.text_input("Label (optional)")

    # Advanced display controls
    st.sidebar.subheader("Display Options")
    grid = st.sidebar.checkbox("Show Grid", value=True)
    color_scheme = st.sidebar.selectbox("Color Scheme", ["default", "viridis", "plasma", "cividis", "magma", "inferno", "tab10", "Set1", "Pastel1"], index=0)
    legend_mode = st.sidebar.radio("Legend Mode", ["Corner", "Manual"], index=0, horizontal=True)
    if legend_mode == "Corner":
        legend_loc = st.sidebar.selectbox("Legend Corner", ["upper right", "upper left", "lower left", "lower right"], index=0)
        legend_xy = None
    else:
        lx = st.sidebar.slider("Legend X", 0.0, 1.0, 0.95, 0.01)
        ly = st.sidebar.slider("Legend Y", 0.0, 1.0, 0.95, 0.01)
        legend_loc = "center"
        legend_xy = (lx, ly)
    legend_draggable = st.sidebar.checkbox("Draggable Legend", value=False)

    # Rate-only parameters
    if plot_type == "Rate":
        theoretical_capacity = st.sidebar.number_input("Theoretical Capacity (mAh/g)", min_value=0.0, value=1675.0)
        cycle_range_text = st.sidebar.text_input("Rate Cycle Range (start-end)", value="1-3")
        try:
            start_s, end_s = [int(x.strip()) for x in cycle_range_text.replace('to', '-').split('-')]
        except Exception:
            start_s, end_s = 1, 3
        c_rate_label_mode = st.sidebar.selectbox(
            "C-rate labels",
            ["No", "Over all plots", "On top"],
            index=1,
            help="Choose how C-rate annotations appear: none, near each segment, or in a line at the top."
        )
    else:
        theoretical_capacity = None
        start_s = end_s = None
        c_rate_label_mode = "No"  # default placeholder when not Rate plot

    # Saving options
    st.sidebar.subheader("Export")
    save_svg = st.sidebar.checkbox("Save SVG")
    svg_name = st.sidebar.text_input("SVG Filename (no ext)", value="") if save_svg else ""
    svg_transparent = st.sidebar.checkbox("SVG Transparent", value=True) if save_svg else True

    save_png = st.sidebar.checkbox("Save PNG")
    png_name = st.sidebar.text_input("PNG Filename (no ext)", value="") if save_png else ""
    png_dpi = st.sidebar.number_input("PNG DPI", min_value=72, max_value=1200, value=300) if save_png else 300
    png_transparent = st.sidebar.checkbox("PNG Transparent", value=True) if save_png else True

    plot_button = st.sidebar.button("Plot")

    placeholder = st.empty()

    if plot_button:
        if not sample or not selected_files:
            st.warning("Please select sample and file(s).")
            return
        group_data, masses = load_group_data(project_folder, sample, selected_files, cycles)
        groups = [{"label": label or sample, "data": group_data, "masses": masses}]

        if not groups[0]["data"]:
            st.error("No valid data loaded.")
            return

        try:
            if plot_type == "GCD":
                if gcd_plotter is None:
                    st.error("GCD plotter not available.")
                    return
                fig, ax = gcd_plotter.plot_gcd_curves(groups, grid=grid, color_scheme=color_scheme, legend_loc=legend_loc, legend_xy=legend_xy, legend_draggable=legend_draggable)
            elif plot_type == "Rate":
                if rate_plotter is None:
                    st.error("Rate plotter not available.")
                    return
                start_cycle = start_s or 1
                end_cycle = end_s or start_cycle
                # Translate UI selection to internal mode keywords
                mode_map = {
                    "No": "none",
                    "Over all plots": "overall",
                    "On top": "on_top",
                }
                # c_rate_label_mode guaranteed non-None by assignment above
                internal_mode = mode_map.get(str(c_rate_label_mode), "overall")
                fig, ax = rate_plotter.plot_rate_performance(
                    groups,
                    theoretical_capacity or 1675.0,
                    start_cycle,
                    end_cycle,
                    grid=grid,
                    color_scheme=color_scheme,
                    legend_loc=legend_loc,
                    legend_xy=legend_xy,
                    legend_draggable=legend_draggable,
                    c_rate_label_mode=internal_mode,
                )
            else:
                st.info("CV plotting not implemented yet.")
                return

            placeholder.pyplot(fig)

            save_dir = ensure_fig_dir(project_folder)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if save_svg:
                base_name = (svg_name.strip() or f"{plot_type}_{timestamp}") + '.svg'
                out_path = unique_path(save_dir, base_name)
                try:
                    fig.savefig(out_path, format='svg', bbox_inches='tight', transparent=svg_transparent)
                    st.success(f"Saved SVG: {out_path}")
                except Exception as e:
                    st.error(f"Failed to save SVG: {e}")
            if save_png:
                base_name = (png_name.strip() or f"{plot_type}_{timestamp}") + '.png'
                out_path = unique_path(save_dir, base_name)
                try:
                    fig.savefig(out_path, format='png', dpi=int(png_dpi), bbox_inches='tight', transparent=png_transparent)
                    st.success(f"Saved PNG: {out_path}")
                except Exception as e:
                    st.error(f"Failed to save PNG: {e}")
        except Exception as e:
            st.error(f"Plotting failed: {e}")

    st.markdown("---")
    st.caption("Generated by Streamlit app wrapper - experimental")

if __name__ == '__main__':
    # Detect if running under 'streamlit run' by checking if ScriptRunContext exists
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None
    if ctx is None:
        print("[INFO] Direct execution detected (bare mode). Use 'streamlit run app/app.py' for full UI.")
    main()
