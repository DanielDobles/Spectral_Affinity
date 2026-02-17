"""Fix over-escaped backslashes in Spectral_Affinity_Master.ipynb."""
import json, os

path = os.path.join(os.path.dirname(__file__), "Spectral_Affinity_Master.ipynb")
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    new_source = []
    for line in cell["source"]:
        # Fix the over-escaped f-string for output name
        line = line.replace(
            r'                on = f\\\"',
            '                on = f"'
        )
        line = line.replace(
            r'.flac\\\"',
            '.flac"'
        )
        # Fix the over-escaped display(HTML(...))  lines
        line = line.replace(
            r'display(HTML(\\\"',
            "display(HTML('"
        )
        line = line.replace(
            r'\\\"))',
            "'))"
        )
        new_source.append(line)
    cell["source"] = new_source

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print(f"✅ Fixed: {path}")
