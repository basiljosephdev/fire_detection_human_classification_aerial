import json
from pathlib import Path

def update_notebook_weight_search():
    for nb_filename in ["Untitled.ipynb", "Untitled1.ipynb"]:
        nb_path = Path(nb_filename)
        if not nb_path.exists():
            continue
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                lines = cell.get("source", [])
                new_lines = []
                for line in lines:
                    if 'posture_pt = project_dir / "models/training/runs/custom_posture_detector/train/weights/best.pt"' in line:
                        line = (
                            'posture_pt_candidates = [\n'
                            '    project_dir / "runs/detect/runs/custom_posture_detector/train-3/weights/best.pt",\n'
                            '    project_dir / "models/training/runs/custom_posture_detector/train/weights/best.pt",\n'
                            '    project_dir / "runs/detect/runs/custom_posture_detector/train/weights/best.pt",\n'
                            '    project_dir / "runs/custom_posture_detector/train/weights/best.pt",\n'
                            ']\n'
                            'posture_pt = next((p for p in posture_pt_candidates if p.exists()), posture_pt_candidates[0])\n'
                        )
                    elif 'project_dir / "models/training/runs/custom_posture_detector/train/weights/best.pt",' in line:
                        new_lines.append(line)
                        new_lines.append('    project_dir / "runs/detect/runs/custom_posture_detector/train/weights/best.pt",\n')
                        continue
                    new_lines.append(line)
                cell["source"] = new_lines

        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {nb_filename} weight search paths")

if __name__ == "__main__":
    update_notebook_weight_search()
