import json

with open("Untitled.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb.get('cells', []))}")
for i, c in enumerate(nb.get("cells", [])):
    print(f"=== Cell {i} ({c.get('cell_type')}) ===")
    src = "".join(c.get("source", []))
    print(src)
    print("-" * 50)
