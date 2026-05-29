import json

with open("e:/Github_repositories/SAXS_WAXS/Final_Notebooks/main_analysis.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        for i, line in enumerate(source):
            if "save_plot=False\n" in line:
                source[i] = line.replace("save_plot=False\n", "save_plot=False,\n")
                source.insert(i+1, "    discharge_files=ec_results.get('discharge_saxs_files', None)\n")
                break

with open("e:/Github_repositories/SAXS_WAXS/Final_Notebooks/main_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
