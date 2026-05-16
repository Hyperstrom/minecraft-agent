# Cell 1 — Install (run this first in Kaggle)
# %%
import subprocess
subprocess.run(["pip","install","-q","unsloth","trl","peft","bitsandbytes",
                "datasets","accelerate","huggingface_hub","minedojo","mcwiki",
                "mwclient","requests"], check=False)

# Cell 2 — GPU check
# %%
import torch
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory/1024**3,1),"GB")
print("BF16:", torch.cuda.is_bf16_supported())
print("CUDA:", torch.version.cuda)
