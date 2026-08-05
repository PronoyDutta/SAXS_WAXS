import sys

with open(r'e:\Github_repositories\SAXS_WAXS\classes_and_functions\utils.py', 'r', encoding='utf-8') as f:
    content = f.read()

robust_extraction_contourplot = """
    # ---- Robust Timestamp Extraction ----
    keys = list(new_time_dict.keys())
    if len(keys) > 0 and isinstance(new_time_dict[keys[0]], (int, np.integer, float)):
        file_vals = list(new_time_dict.values())
        if all(isinstance(v, (int, np.integer)) for v in file_vals) and min(file_vals) >= 1:
            file_to_time = {v: float(k) for k, v in new_time_dict.items()}
            base_time = file_to_time.get(starting_file, 0.0)
            get_elapsed = lambda f: file_to_time.get(f, 0.0) - base_time
        else:
            def parse_ts(v):
                if isinstance(v, (int, float, np.number)): return float(v)
                if isinstance(v, bytes): v = v.decode('utf-8')
                try: return datetime.strptime(str(v), '%Y-%m-%dT%H:%M:%S').timestamp()
                except: return float(v)
            base_val = new_time_dict.get(starting_file)
            base_time = parse_ts(base_val) if base_val is not None else 0.0
            get_elapsed = lambda f: parse_ts(new_time_dict.get(f)) - base_time if new_time_dict.get(f) is not None else 0.0
    else:
        def parse_ts(v):
            if isinstance(v, (int, float, np.number)): return float(v)
            if isinstance(v, bytes): v = v.decode('utf-8')
            try: return datetime.strptime(str(v), '%Y-%m-%dT%H:%M:%S').timestamp()
            except: return float(v)
        base_val = new_time_dict.get(starting_file)
        base_time = parse_ts(base_val) if base_val is not None else 0.0
        get_elapsed = lambda f: parse_ts(new_time_dict.get(f)) - base_time if new_time_dict.get(f) is not None else 0.0

    # Process data matrices
    for file_number in range(starting_file, last_file + 1):
        if data_normalization == 'relative' and Normalization_file is not None:
            norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1] / data_dictionary[Normalization_file].iloc[start_row:end_row + 1, 1]
        else:
            norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1]

        Intensity_list.append(norm_intensity.to_numpy())
        time_t1.append(get_elapsed(file_number))
"""

old_block_contourplot = """
    # Check if new_time_dict contains raw timestamps (strings/bytes) or elapsed seconds (floats)
    is_raw_timestamps = isinstance(keys[0], (int, np.integer)) and isinstance(new_time_dict[keys[0]], (str, bytes))
    
    if is_raw_timestamps:
        val0 = new_time_dict[starting_file]
        if isinstance(val0, bytes): val0 = val0.decode('utf-8')
        base_time = datetime.strptime(str(val0), '%Y-%m-%dT%H:%M:%S')

    # Process data matrices
    for file_number in range(starting_file, last_file + 1):
        if data_normalization == 'relative' and Normalization_file is not None:
            norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1] / data_dictionary[Normalization_file].iloc[start_row:end_row + 1, 1]
        else:
            norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1]

        Intensity_list.append(norm_intensity.to_numpy())
        
        # Calculate time elapsed
        if is_raw_timestamps:
            val = new_time_dict[file_number]
            if isinstance(val, bytes): val = val.decode('utf-8')
            current_time = datetime.strptime(str(val), '%Y-%m-%dT%H:%M:%S')
            time_elapsed = (current_time - base_time).total_seconds()
        else:
            # Legacy format where keys are ordered elapsed seconds
            time_elapsed = float(keys[file_number - 1] - keys[starting_file - 1])
            
        time_t1.append(time_elapsed)
"""

if old_block_contourplot.strip() in content.strip():
    print("Found contourplot block")
content = content.replace(old_block_contourplot.strip(), robust_extraction_contourplot.strip())


robust_extraction_combined = """
    # ---- Robust Timestamp Extraction ----
    keys = list(new_time_dict.keys())
    if len(keys) > 0 and isinstance(new_time_dict[keys[0]], (int, np.integer, float)):
        file_vals = list(new_time_dict.values())
        if all(isinstance(v, (int, np.integer)) for v in file_vals) and min(file_vals) >= 1:
            file_to_time = {v: float(k) for k, v in new_time_dict.items()}
            base_time = file_to_time.get(starting_file, 0.0)
            get_elapsed = lambda f: file_to_time.get(f, 0.0) - base_time
        else:
            def parse_ts(v):
                if isinstance(v, (int, float, np.number)): return float(v)
                if isinstance(v, bytes): v = v.decode('utf-8')
                try: return datetime.strptime(str(v), '%Y-%m-%dT%H:%M:%S').timestamp()
                except: return float(v)
            base_val = new_time_dict.get(starting_file)
            base_time = parse_ts(base_val) if base_val is not None else 0.0
            get_elapsed = lambda f: parse_ts(new_time_dict.get(f)) - base_time if new_time_dict.get(f) is not None else 0.0
    else:
        def parse_ts(v):
            if isinstance(v, (int, float, np.number)): return float(v)
            if isinstance(v, bytes): v = v.decode('utf-8')
            try: return datetime.strptime(str(v), '%Y-%m-%dT%H:%M:%S').timestamp()
            except: return float(v)
        base_val = new_time_dict.get(starting_file)
        base_time = parse_ts(base_val) if base_val is not None else 0.0
        get_elapsed = lambda f: parse_ts(new_time_dict.get(f)) - base_time if new_time_dict.get(f) is not None else 0.0

    for file_number in range(starting_file, last_file + 1):
        time_t1.append(get_elapsed(file_number))
"""

old_block_combined = """
    is_raw_timestamps = isinstance(keys[0], (int, np.integer)) and isinstance(new_time_dict[keys[0]], (str, bytes))
    if is_raw_timestamps:
        val0 = new_time_dict[starting_file]
        if isinstance(val0, bytes): val0 = val0.decode('utf-8')
        base_time = datetime.strptime(str(val0), '%Y-%m-%dT%H:%M:%S')

    def get_grid(data_dictionary, min_limit, max_limit, normalization_type, is_waxs=False, norm_file=None):
        X1_data_temp = 10 * data_dictionary[starting_file].iloc[:, 0]
        if is_waxs:
            X1_data_temp = 2 * np.degrees(np.arcsin((wavelength_nm * X1_data_temp) / (4 * np.pi)))
            
        indices = np.where((X1_data_temp >= min_limit) & (X1_data_temp <= max_limit))
        X1_filtered = X1_data_temp.iloc[indices]
        start_row, end_row = np.min(indices), np.max(indices)
        
        Intensity_list = []
        for file_number in range(starting_file, last_file + 1):
            if normalization_type == 'relative' and norm_file is not None:
                norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1] / data_dictionary[norm_file].iloc[start_row:end_row + 1, 1]
            else:
                norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1]
            Intensity_list.append(norm_intensity.to_numpy())
            
        normIntensity_array = np.array(Intensity_list)
        return X1_filtered, gaussian_filter(normIntensity_array, sigma=1)

    X_saxs, Z_saxs = get_grid(saxs_dict, saxs_min_limit, saxs_max_limit, saxs_data_normalization, is_waxs=False, norm_file=Normalization_file_saxs)
    X_waxs, Z_waxs = get_grid(waxs_dict, waxs_min_limit, waxs_max_limit, waxs_data_normalization, is_waxs=True, norm_file=Normalization_file_waxs)
    
    for file_number in range(starting_file, last_file + 1):
        if is_raw_timestamps:
            val = new_time_dict[file_number]
            if isinstance(val, bytes): val = val.decode('utf-8')
            current_time = datetime.strptime(str(val), '%Y-%m-%dT%H:%M:%S')
            time_t1.append((current_time - base_time).total_seconds())
        else:
            time_t1.append(float(keys[file_number - 1] - keys[starting_file - 1]))
"""

new_block_combined_full = """
    def get_grid(data_dictionary, min_limit, max_limit, normalization_type, is_waxs=False, norm_file=None):
        X1_data_temp = 10 * data_dictionary[starting_file].iloc[:, 0]
        if is_waxs:
            X1_data_temp = 2 * np.degrees(np.arcsin((wavelength_nm * X1_data_temp) / (4 * np.pi)))
            
        indices = np.where((X1_data_temp >= min_limit) & (X1_data_temp <= max_limit))
        X1_filtered = X1_data_temp.iloc[indices]
        start_row, end_row = np.min(indices), np.max(indices)
        
        Intensity_list = []
        for file_number in range(starting_file, last_file + 1):
            if normalization_type == 'relative' and norm_file is not None:
                norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1] / data_dictionary[norm_file].iloc[start_row:end_row + 1, 1]
            else:
                norm_intensity = data_dictionary[file_number].iloc[start_row:end_row + 1, 1]
            Intensity_list.append(norm_intensity.to_numpy())
            
        normIntensity_array = np.array(Intensity_list)
        return X1_filtered, gaussian_filter(normIntensity_array, sigma=1)

    X_saxs, Z_saxs = get_grid(saxs_dict, saxs_min_limit, saxs_max_limit, saxs_data_normalization, is_waxs=False, norm_file=Normalization_file_saxs)
    X_waxs, Z_waxs = get_grid(waxs_dict, waxs_min_limit, waxs_max_limit, waxs_data_normalization, is_waxs=True, norm_file=Normalization_file_waxs)
    
""" + robust_extraction_combined

if old_block_combined.strip() in content.strip():
    print("Found combined_contourplot block")
content = content.replace(old_block_combined.strip(), new_block_combined_full.strip())

with open(r'e:\Github_repositories\SAXS_WAXS\classes_and_functions\utils.py', 'w', encoding='utf-8') as f:
    f.write(content)

import json
with open(r'e:\Github_repositories\SAXS_WAXS\Final_Notebooks\main_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'combined_contourplot(' in ''.join(cell.get('source', [])):
        src = ''.join(cell['source'])
        src = src.replace("# applied_current=0.5", "applied_current=ec_results['applied_current']")
        src = src.replace("# active_material_mass=2.0", "active_material_mass=ec_results['active_material_mass']")
        cell['source'] = src.splitlines(True)

with open(r'e:\Github_repositories\SAXS_WAXS\Final_Notebooks\main_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
