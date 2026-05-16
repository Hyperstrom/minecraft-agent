"""
restore_cell4.py
"""
import json
from pathlib import Path

nb_path = Path("training/mineagent_train.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

CELL_4_SOURCE = [
    "# ── Cell 4: Train Stage 1 (Knowledge) ────────────────────────────\n",
    "from unsloth import FastLanguageModel\n",
    "from trl import SFTTrainer\n",
    "from transformers import TrainingArguments\n",
    "from datasets import load_dataset\n",
    "\n",
    "MODEL_ID = 'unsloth/Llama-3.2-3B-Instruct'\n",
    "\n",
    "model, tok = FastLanguageModel.from_pretrained(\n",
    "    model_name=MODEL_ID, max_seq_length=512,\n",
    "    dtype=None, load_in_4bit=True)\n",
    "\n",
    "model = FastLanguageModel.get_peft_model(\n",
    "    model, r=16, lora_alpha=32,\n",
    "    target_modules=['q_proj','k_proj','v_proj','o_proj'],\n",
    "    lora_dropout=0.05, bias='none')\n",
    "\n",
    "tok.pad_token = tok.eos_token\n",
    "tok.padding_side = 'right'\n",
    "\n",
    "ds1 = load_dataset('json', data_files={'train':'/kaggle/working/data/stage1.jsonl'}, split='train')\n",
    "split1 = ds1.train_test_split(test_size=0.05, seed=42)\n",
    "\n",
    "def fmt(ex): return {'text': tok.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=False)}\n",
    "split1 = split1.map(fmt)\n",
    "\n",
    "trainer1 = SFTTrainer(\n",
    "    model=model, tokenizer=tok,\n",
    "    train_dataset=split1['train'], eval_dataset=split1['test'],\n",
    "    dataset_text_field='text', max_seq_length=512,\n",
    "    args=TrainingArguments(\n",
    "        output_dir='/kaggle/working/stage1',\n",
    "        num_train_epochs=1,\n",
    "        per_device_train_batch_size=8,\n",
    "        gradient_accumulation_steps=4,\n",
    "        learning_rate=3e-4,\n",
    "        lr_scheduler_type='cosine',\n",
    "        warmup_ratio=0.05,\n",
    "        fp16=True, bf16=False, logging_steps=50,\n",
    "        eval_strategy='steps', eval_steps=300,\n",
    "        save_steps=500, save_total_limit=1,\n",
    "        report_to='none',\n",
    "    ))\n",
    "\n",
    "print('=== Stage 1 Training Start ===')\n",
    "trainer1.train()\n",
    "model.save_pretrained('/kaggle/working/stage1-lora')\n",
    "tok.save_pretrained('/kaggle/working/stage1-lora')\n"
]

for cell in nb["cells"]:
    if cell.get("id") == "c5":
        cell["source"] = CELL_4_SOURCE
        print("Restored Cell 4 (id c5)")

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved:", nb_path)
