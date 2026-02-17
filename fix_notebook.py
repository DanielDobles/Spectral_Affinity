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
        # Fix the f-string for output name (on = f"...")
        line = line.replace('f\\"', 'f"')
        line = line.replace('.flac\\"', '.flac"')
        
        # Fix the display(HTML(...)) lines
        line = line.replace('HTML(\\"', 'HTML("')
        line = line.replace('\\"))', '"))')
        
        # Also fix any remaining triple-escaped quotes if they exist
        line = line.replace('\\\\\\"', '"')
        
        new_source.append(line)
    cell["source"] = new_source

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print(f"✅ Fixed: {path}")
