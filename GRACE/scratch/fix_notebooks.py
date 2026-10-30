import json
from pathlib import Path

def fix_notebook_1():
    nb_path = Path("Untitled1.ipynb")
    if not nb_path.exists():
        print("Untitled1.ipynb not found")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            lines = cell.get("source", [])
            new_lines = []
            for line in lines:
                if "posture_model.predict(" in line and "classes=" not in line:
                    line = line.replace(
                        "posture_model.predict(source=str(img_path), conf=0.25)",
                        "posture_model.predict(source=str(img_path), conf=0.35, classes=[0] if posture_model_path == 'yolov8n.pt' else None)"
                    )
                new_lines.append(line)
            cell["source"] = new_lines

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated Untitled1.ipynb successfully")

def fix_notebook_0():
    nb_path = Path("Untitled.ipynb")
    if not nb_path.exists():
        print("Untitled.ipynb not found")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            lines = cell.get("source", [])
            new_lines = []
            for line in lines:
                if "posture_model.predict(" in line and "classes=" not in line:
                    line = line.replace(
                        "posture_model.predict(source=str(img_path), conf=0.25)",
                        "posture_model.predict(source=str(img_path), conf=0.35, classes=[0] if posture_model_path == 'yolov8n.pt' else None)"
                    )
                elif "results = posture_model.predict(" in line and "classes=" not in line:
                    line = line.replace(
                        "results = posture_model.predict(source=str(img_path), conf=0.25)",
                        "results = posture_model.predict(source=str(img_path), conf=0.35, classes=[0] if posture_model_path == 'yolov8n.pt' else None)"
                    )
                new_lines.append(line)
            cell["source"] = new_lines

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Updated Untitled.ipynb successfully")

if __name__ == "__main__":
    fix_notebook_1()
    fix_notebook_0()
