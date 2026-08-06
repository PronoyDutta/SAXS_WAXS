import ipywidgets as widgets
from IPython.display import display
import copy
import numpy as np
import matplotlib.pyplot as plt
from pybaselines import Baseline

def create_masking_ui(base_waxs, output_dict):
    """
    Creates an interactive UI for masking sharp peaks in WAXS data.
    
    Parameters:
    -----------
    base_waxs : dict
        The original input dictionary of pandas DataFrames (e.g., all_data['WAXS']).
    output_dict : dict
        An empty dictionary passed from the notebook. It will be populated with the
        masked data when the user clicks 'Apply'.
    """
    
    # Initialize with a copy so if no masking is done, the output is identical
    output_dict.clear()
    output_dict.update(copy.deepcopy(base_waxs))
    
    mask_regions_input = widgets.Textarea(
        value='',
        placeholder='Enter regions to mask, e.g.:\n1.94, 1.98\n2.14, 2.16',
        description='Mask Regions:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(height='100px', width='300px')
    )
    
    ref_file_mask_input = widgets.Dropdown(
        options=list(base_waxs.keys()),
        description='Ref File:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='150px')
    )
    
    x_min_plot = widgets.FloatText(
        value=1.4,
        description='Plot X Min:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='150px')
    )
    
    x_max_plot = widgets.FloatText(
        value=2.5,
        description='Plot X Max:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='150px')
    )
    
    preview_mask_btn = widgets.Button(description='Preview Mask', button_style='info')
    apply_mask_btn = widgets.Button(description='Apply to All', button_style='success')
    output_mask = widgets.Output()
    
    def parse_mask_regions(text):
        regions = []
        for line in text.split('\n'):
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            if len(parts) == 2:
                try:
                    regions.append((float(parts[0].strip()), float(parts[1].strip())))
                except ValueError:
                    pass
        return regions
    
    def apply_mask_to_df(df, regions):
        df_masked = df.copy()
        for qmin, qmax in regions:
            mask = (df_masked.iloc[:, 0] >= qmin) & (df_masked.iloc[:, 0] <= qmax)
            df_masked.loc[mask, df_masked.columns[1]] = np.nan
        df_masked.iloc[:, 1] = df_masked.iloc[:, 1].interpolate(method='linear')
        return df_masked
    
    def on_preview_mask(b):
        with output_mask:
            output_mask.clear_output()
            regions = parse_mask_regions(mask_regions_input.value)
            
            ref_idx = ref_file_mask_input.value
            original_df = base_waxs[ref_idx]
            if regions:
                masked_df = apply_mask_to_df(original_df, regions)
            else:
                masked_df = original_df
            
            plt.figure(figsize=(10, 5))
            plt.plot(original_df.iloc[:, 0], original_df.iloc[:, 1], label='Original', alpha=0.7)
            if regions:
                plt.plot(masked_df.iloc[:, 0], masked_df.iloc[:, 1], label='Masked & Interpolated', linestyle='--')
                for qmin, qmax in regions:
                    plt.axvspan(qmin, qmax, color='red', alpha=0.2, label='Masked Region' if regions.index((qmin, qmax)) == 0 else "")
                
            # Use by_label logic for duplicate labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            
            plt.title(f"Peak Masking Preview (File {ref_idx})")
            plt.xlabel("Q (Å⁻¹)")
            plt.ylabel("Intensity")
            
            # Apply X limits if valid
            if x_min_plot.value < x_max_plot.value:
                plt.xlim(x_min_plot.value, x_max_plot.value)
                
                # Autoscale Y axis based on the visible X range
                mask = (original_df.iloc[:, 0] >= x_min_plot.value) & (original_df.iloc[:, 0] <= x_max_plot.value)
                visible_y = original_df.loc[mask].iloc[:, 1]
                if not visible_y.empty:
                    y_min = visible_y.min()
                    y_max = visible_y.max()
                    y_margin = (y_max - y_min) * 0.1
                    if y_margin == 0: y_margin = 0.1
                    plt.ylim(y_min - y_margin, y_max + y_margin)
                    
            plt.show()
    
    def on_apply_mask(b):
        with output_mask:
            output_mask.clear_output()
            regions = parse_mask_regions(mask_regions_input.value)
            output_dict.clear()
            if not regions:
                print("No valid regions to apply. Passing original data directly to 'waxs_dict_to_correct'.")
                output_dict.update(copy.deepcopy(base_waxs))
                return
                
            print(f"Applying mask for regions {regions} to all {len(base_waxs)} files...")
            for k, df in base_waxs.items():
                output_dict[k] = apply_mask_to_df(df, regions)
                
            print("Masking complete! Data saved successfully in dictionary: 'waxs_dict_to_correct'")
            print("You can now proceed to the Baseline Correction step.")
    
    preview_mask_btn.on_click(on_preview_mask)
    apply_mask_btn.on_click(on_apply_mask)
    
    form_mask = widgets.VBox([
        widgets.HTML("<h3>WAXS Peak Masking (Optional)</h3>"),
        widgets.HTML("<p>Remove sharp artifact peaks by linearly interpolating across them.</p>"),
        widgets.HBox([mask_regions_input, widgets.VBox([ref_file_mask_input, x_min_plot, x_max_plot])]),
        widgets.HBox([preview_mask_btn, apply_mask_btn]),
        output_mask
    ])
    return form_mask

def create_baseline_ui(waxs_dict_to_correct, output_dict):
    """
    Creates an interactive UI for applying baseline correction.
    
    Parameters:
    -----------
    waxs_dict_to_correct : dict
        The input dictionary to be baseline corrected (either raw or masked).
    output_dict : dict
        An empty dictionary passed from the notebook. It will be populated with the
        baseline-corrected data when the user clicks 'Apply'.
    """
    
    # Initialize with a copy
    output_dict.clear()
    output_dict.update(copy.deepcopy(waxs_dict_to_correct))
    
    baseline_mode_dropdown = widgets.Dropdown(
        options=['None', 'Single Baseline', 'Individual Baselines'],
        value='None',
        description='Mode:',
        style={'description_width': 'initial'}
    )
    
    ref_file_input = widgets.Dropdown(
        options=list(waxs_dict_to_correct.keys()),
        description='Ref File:',
        style={'description_width': 'initial'}
    )
    
    poly_order_input = widgets.IntText(
        value=3,
        description='Poly Order:',
        style={'description_width': 'initial'}
    )
    
    x_min_input = widgets.FloatText(
        value=1.4,
        description='Crop X Min (Å$^{-1}$):',
        style={'description_width': 'initial'}
    )
    
    x_max_input = widgets.FloatText(
        value=2.5,
        description='Crop X Max:',
        style={'description_width': 'initial'}
    )
    
    preview_btn = widgets.Button(description='Preview Baseline', button_style='info')
    apply_btn = widgets.Button(description='Apply Correction', button_style='success')
    output_baseline = widgets.Output()
    
    def get_crop_indices(x_data, x_min, x_max):
        if np.any(x_data <= x_min):
            idx_min = np.max(np.where(x_data <= x_min))
        else:
            idx_min = 0
            
        if np.any(x_data <= x_max):
            idx_max = np.max(np.where(x_data <= x_max))
        else:
            idx_max = len(x_data)
            
        if idx_max <= idx_min:
            idx_min = 0
            idx_max = len(x_data)
            
        return idx_min, idx_max
    
    def on_preview_clicked(b):
        with output_baseline:
            output_baseline.clear_output()
            ref_idx = ref_file_input.value
            df = waxs_dict_to_correct[ref_idx]
            x_data = df.iloc[:, 0].to_numpy()
            y_data = df.iloc[:, 1].to_numpy()
            
            idx_min, idx_max = get_crop_indices(x_data, x_min_input.value, x_max_input.value)
            x_fit = x_data[idx_min:idx_max]
            y_fit = y_data[idx_min:idx_max]
            
            baseline_filter = Baseline(x_data=x_fit)
            baseline = baseline_filter.modpoly(y_fit, poly_order=poly_order_input.value)[0]
            
            plt.figure(figsize=(8, 4))
            plt.plot(x_fit, y_fit, label='Cropped Original Data')
            plt.plot(x_fit, baseline, label='Baseline', linestyle='--')
            plt.legend()
            plt.title(f"Baseline fit for file {ref_idx}")
            plt.show()
            
    def on_apply_clicked(b):
        with output_baseline:
            output_baseline.clear_output()
            mode = baseline_mode_dropdown.value
            output_dict.clear()
            
            if mode == 'None':
                output_dict.update(copy.deepcopy(waxs_dict_to_correct))
                print("Baseline correction disabled. Original data saved into dictionary: 'waxs_baseline_corrected_dict'")
                return
            
            print(f"Applying {mode} correction with cropping...")
            
            ref_idx = ref_file_input.value
            ref_df = waxs_dict_to_correct[ref_idx]
            x_ref = ref_df.iloc[:, 0].to_numpy()
            y_ref = ref_df.iloc[:, 1].to_numpy()
            
            idx_min_ref, idx_max_ref = get_crop_indices(x_ref, x_min_input.value, x_max_input.value)
            x_ref_fit = x_ref[idx_min_ref:idx_max_ref]
            y_ref_fit = y_ref[idx_min_ref:idx_max_ref]
            
            baseline_filter = Baseline(x_data=x_ref_fit)
            single_baseline = baseline_filter.modpoly(y_ref_fit, poly_order=poly_order_input.value)[0]
            
            for k, df in waxs_dict_to_correct.items():
                x_data = df.iloc[:, 0].to_numpy()
                y_data = df.iloc[:, 1].to_numpy()
                
                idx_min, idx_max = get_crop_indices(x_data, x_min_input.value, x_max_input.value)
                x_fit = x_data[idx_min:idx_max]
                y_fit = y_data[idx_min:idx_max]
                
                if mode == 'Single Baseline':
                    b_line = single_baseline
                else:
                    b_filter = Baseline(x_data=x_fit)
                    b_line = b_filter.modpoly(y_fit, poly_order=poly_order_input.value)[0]
                
                new_df = df.iloc[idx_min:idx_max].copy()
                new_df.iloc[:, 1] = y_fit - b_line
                output_dict[k] = new_df
            
            print(f"Successfully corrected and cropped {len(output_dict)} WAXS files!")
            print("Data saved successfully in dictionary: 'waxs_baseline_corrected_dict'")
            
    preview_btn.on_click(on_preview_clicked)
    apply_btn.on_click(on_apply_clicked)
    
    form = widgets.VBox([
        widgets.HTML("<h3>WAXS Baseline Correction</h3>"),
        baseline_mode_dropdown,
        ref_file_input,
        poly_order_input,
        widgets.HBox([x_min_input, x_max_input]),
        widgets.HBox([preview_btn, apply_btn]),
        output_baseline
    ])
    return form
