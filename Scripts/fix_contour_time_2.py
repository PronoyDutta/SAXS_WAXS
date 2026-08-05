import sys

with open(r'e:\Github_repositories\SAXS_WAXS\classes_and_functions\utils.py', 'r', encoding='utf-8') as f:
    content = f.read()

def replace_block(content, old_str, new_str):
    if old_str in content:
        return content.replace(old_str, new_str)
    return content

old_get_elapsed_1 = """
            base_val = new_time_dict.get(starting_file)
            base_time = parse_ts(base_val) if base_val is not None else 0.0
            get_elapsed = lambda f: parse_ts(new_time_dict.get(f)) - base_time if new_time_dict.get(f) is not None else 0.0
"""

new_get_elapsed_1 = """
            def get_val(d, k):
                if k in d: return d[k]
                if str(k) in d: return d[str(k)]
                try: return d[int(k)]
                except: return None
            
            base_val = get_val(new_time_dict, starting_file)
            base_time = parse_ts(base_val) if base_val is not None else 0.0
            get_elapsed = lambda f: parse_ts(get_val(new_time_dict, f)) - base_time if get_val(new_time_dict, f) is not None else 0.0
"""

content = replace_block(content, old_get_elapsed_1, new_get_elapsed_1)

with open(r'e:\Github_repositories\SAXS_WAXS\classes_and_functions\utils.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied.")
