import json

with open(r'e:\Github_repositories\SAXS_WAXS\Final_Notebooks\main_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'combined_contourplot(' in ''.join(cell.get('source', [])):
        src = ''.join(cell['source'])
        src = src.replace("y_axis_mode='capacity'", "y_axis_mode='time'")
        cell['source'] = src.splitlines(True)

with open(r'e:\Github_repositories\SAXS_WAXS\Final_Notebooks\main_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook updated to use y_axis_mode='time'")
