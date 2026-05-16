"""
patch_notebook_warnings.py
"""
import json
from pathlib import Path

nb_path = Path("training/mineagent_train.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell.get("id") == "c2":  # Cell 1 - Pip installs
        source = "".join(cell["source"])
        if "os.environ['TF_CPP_MIN_LOG_LEVEL']" not in source:
            source = "import os\nos.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\nos.environ['WARNINGS_LOG'] = '0'\nimport warnings\nwarnings.filterwarnings('ignore')\n\n" + source
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")[:-1]]
            print("Patched global warnings")

    if cell.get("id") == "c3":  # Cell 2 - GPU login
        source = "".join(cell["source"])
        if "unsloth" not in source:
            # Import unsloth BEFORE torch
            source = "import unsloth # Import early to satisfy unsloth warnings\n" + source
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")[:-1]]
            print("Patched Cell 2 imports")

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved:", nb_path)
