"""
patch_notebook_fast2.py
"""
import json
from pathlib import Path

nb_path = Path("training/mineagent_train.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell.get("id") == "c4":  # Data collection
        source = "".join(cell["source"])
        source = source.replace("limit=18000", "limit=8000") # Reduce data size from 27k to ~12k
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")[:-1]]
        print("Patched Data Limit")

    if cell.get("id") == "c6":  # Stage 2
        source = "".join(cell["source"])
        source = source.replace("num_train_epochs=5", "num_train_epochs=3") # Reduce epochs
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")[:-1]]
        print("Patched Stage 2 Epochs")

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved:", nb_path)
