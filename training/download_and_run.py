"""
download_and_run.py
Run this locally AFTER Kaggle training is complete.

Usage:
  conda activate mineagent-train
  python training/download_and_run.py --hf-user YOUR_HF_USERNAME
"""
import argparse, os, subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--hf-user", required=True, help="Your HuggingFace username")
parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN",""), help="HF token (or set HF_TOKEN env var)")
args = parser.parse_args()

REPO_ID   = f"{args.hf_user}/mineagent-v1"
OUT_DIR   = Path("models/mineagent-v1-gguf")
MODELFILE = Path("training/Modelfile")

# ── Step 1: Download GGUF from HuggingFace ────────────────────────
print(f"\n[1/3] Downloading from https://huggingface.co/{REPO_ID}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id   = REPO_ID,
    local_dir = str(OUT_DIR),
    token     = args.hf_token or None,
)
print(f"Downloaded to: {OUT_DIR}")

# Find the .gguf file
gguf_files = list(OUT_DIR.glob("*.gguf"))
if not gguf_files:
    print("ERROR: No .gguf file found in download!")
    exit(1)

gguf_path = gguf_files[0].resolve()
print(f"GGUF file: {gguf_path}")

# ── Step 2: Create Modelfile ──────────────────────────────────────
print("\n[2/3] Creating Ollama Modelfile...")
modelfile_content = f"""FROM {gguf_path}

PARAMETER temperature 0.1
PARAMETER num_predict 256
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER top_p 0.9

SYSTEM \"\"\"You are MineAgent, an autonomous Minecraft AI agent.
Given the current game state and goal, output ONLY a valid JSON action.
Format: {{"action": "NAME", "params": {{}}, "reasoning": "short reason"}}
Available actions: SEEK, MINE, MOVE, CRAFT, EAT, CHAT, FOLLOW, GOTO, IDLE, STOP
NEVER mine grass_block or dirt unless the goal explicitly requires it.
NEVER output anything outside the JSON object.\"\"\"
"""
MODELFILE.parent.mkdir(exist_ok=True)
MODELFILE.write_text(modelfile_content)
print(f"Modelfile written: {MODELFILE}")

# ── Step 3: Register with Ollama ──────────────────────────────────
print("\n[3/3] Registering model with Ollama...")
result = subprocess.run(
    ["ollama", "create", "mineagent-v1", "-f", str(MODELFILE)],
    capture_output=True, text=True
)
if result.returncode == 0:
    print("Model registered successfully!")
else:
    print("Ollama error:", result.stderr)
    exit(1)

# ── Step 4: Quick test ────────────────────────────────────────────
print("\n=== Quick Test ===")
test = subprocess.run(
    ["ollama", "run", "mineagent-v1",
     'Goal: collect wood. HP:20 Nearby:[oak_log]. Reply with JSON only.'],
    capture_output=True, text=True, timeout=30
)
print("Response:", test.stdout.strip())

# ── Step 5: Update .env ───────────────────────────────────────────
env_path = Path(".env")
if env_path.exists():
    env = env_path.read_text()
    env = "\n".join(
        f"OLLAMA_MODEL=mineagent-v1" if "OLLAMA_MODEL=" in line else line
        for line in env.splitlines()
    )
    if "OLLAMA_MODEL=" not in env:
        env += "\nOLLAMA_MODEL=mineagent-v1\n"
    env_path.write_text(env)
    print("\n.env updated: OLLAMA_MODEL=mineagent-v1")

print("\n✅ Done! Start bot with:")
print("  cd backend && python main.py")
print("  cd bot     && node bot.js")
print("  In game:   !goal collect 10 wood → !planner on")
