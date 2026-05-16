"""
patch_notebook_fp16.py — Fixes BF16 unsupported issue on Kaggle T4
"""
import json
from pathlib import Path

nb_path = Path("training/mineagent_train.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell.get("id") == "c5":  # Cell 4: Train Stage 1
        source = "".join(cell["source"])
        source = source.replace("dtype=torch.bfloat16", "dtype=None")
        source = source.replace("bf16=True", "fp16=True, bf16=False")
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")[:-1]]
        print("Patched Stage 1")

    if cell.get("id") == "c6":  # Cell 5: Train Stage 2
        source = "".join(cell["source"])
        source = source.replace("dtype=torch.bfloat16", "dtype=None")
        source = source.replace("bf16=True", "fp16=True, bf16=False")
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")[:-1]]
        print("Patched Stage 2")

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved:", nb_path)
