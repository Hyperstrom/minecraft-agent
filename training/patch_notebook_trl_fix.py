"""
patch_notebook_trl_fix.py — Fixes Unsloth/TRL compatibility issue
"""
import json
from pathlib import Path

nb_path = Path("training/mineagent_train.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

CELL1_NEW = [
    "import subprocess, sys\n",
    "\n",
    "# Core training libs (Pinning trl<0.9.0 because newer versions break Unsloth's dynamic GKD compilation)\n",
    "subprocess.run([sys.executable,'-m','pip','install','-q',\n",
    "    'unsloth','trl<0.9.0','peft',\n",
    "    'bitsandbytes','datasets>=2.14','accelerate',\n",
    "    'huggingface_hub','mwclient','requests'], check=False)\n",
    "\n",
    "print('Install done')"
]

# Apply patch
for cell in nb["cells"]:
    if cell.get("id") == "c2":
        cell["source"] = CELL1_NEW
        print(f"Patched cell id={cell['id']}")

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved:", nb_path)
