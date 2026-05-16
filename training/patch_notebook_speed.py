"""
patch_notebook_speed.py
"""
import json
from pathlib import Path

nb_path = Path("training/mineagent_train.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell.get("id") == "c5":  # Stage 1
        source = "".join(cell["source"])
        source = source.replace("num_train_epochs=3", "num_train_epochs=1") # 27k pairs is enough for 1 epoch
        source = source.replace("per_device_train_batch_size=4", "per_device_train_batch_size=8") # Use full 15GB VRAM
        source = source.replace("gradient_accumulation_steps=2", "gradient_accumulation_steps=4")
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")[:-1]]
        print("Patched Stage 1 Speed")

    if cell.get("id") == "c6":  # Stage 2
        source = "".join(cell["source"])
        source = source.replace("per_device_train_batch_size=2", "per_device_train_batch_size=4")
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")[:-1]]
        print("Patched Stage 2 Speed")

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved:", nb_path)
