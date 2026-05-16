"""
patch_notebook.py — Patches mineagent_train.ipynb to use synthetic_stage2.jsonl
Run once: python training/patch_notebook.py
"""
import json
from pathlib import Path

nb_path = Path("training/mineagent_train.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

NEW_CELL5_SOURCE = [
    "# ── Cell 5: Train Stage 2 (Behavior Cloning) ─────────────────────\n",
    "import gc\n",
    "del trainer1; gc.collect(); torch.cuda.empty_cache()\n",
    "\n",
    "model2, tok2 = FastLanguageModel.from_pretrained(\n",
    "    model_name='/kaggle/working/stage1-lora',\n",
    "    max_seq_length=1024, dtype=torch.bfloat16, load_in_4bit=True)\n",
    "\n",
    "model2 = FastLanguageModel.get_peft_model(\n",
    "    model2, r=16, lora_alpha=32,\n",
    "    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],\n",
    "    lora_dropout=0.05, bias='none')\n",
    "\n",
    "tok2.pad_token = tok2.eos_token; tok2.padding_side = 'right'\n",
    "\n",
    "# ── Load Stage 2 data ────────────────────────────────────────────\n",
    "# Priority: 1) Bot sessions  2) Synthetic dataset (1500 unique examples)\n",
    "stage2_path = '/kaggle/working/data/stage2.jsonl'\n",
    "synth_path  = '/kaggle/input/mineagent-sessions/synthetic_stage2.jsonl'\n",
    "\n",
    "stage2_size = Path(stage2_path).stat().st_size if Path(stage2_path).exists() else 0\n",
    "\n",
    "if stage2_size < 500:  # no real session data\n",
    "    if Path(synth_path).exists():\n",
    "        import shutil\n",
    "        shutil.copy(synth_path, stage2_path)\n",
    "        lines = open(stage2_path).readlines()\n",
    "        print(f'Loaded {len(lines)} synthetic examples (1500 unique, diverse scenarios)')\n",
    "    else:\n",
    "        print('ERROR: synthetic_stage2.jsonl not found!')\n",
    "        print('Upload training/synthetic_stage2.jsonl to your Kaggle dataset named mineagent-sessions')\n",
    "        raise FileNotFoundError('Missing synthetic_stage2.jsonl')\n",
    "else:\n",
    "    # Merge real sessions + synthetic for best results\n",
    "    if Path(synth_path).exists():\n",
    "        with open(stage2_path, 'a') as dst, open(synth_path) as src:\n",
    "            dst.write(src.read())\n",
    "        print('Merged real sessions + 1500 synthetic examples')\n",
    "    lines = open(stage2_path).readlines()\n",
    "    print(f'Total stage2 examples: {len(lines)}')\n",
    "\n",
    "ds2 = load_dataset('json', data_files={'train': stage2_path}, split='train')\n",
    "split2 = ds2.train_test_split(test_size=0.08, seed=42)\n",
    "split2 = split2.map(lambda x: {'text': tok2.apply_chat_template(\n",
    "    x['messages'], tokenize=False, add_generation_prompt=False)})\n",
    "\n",
    "print(f'Train: {len(split2[\"train\"])} | Val: {len(split2[\"test\"])}')\n",
    "\n",
    "trainer2 = SFTTrainer(\n",
    "    model=model2, tokenizer=tok2,\n",
    "    train_dataset=split2['train'], eval_dataset=split2['test'],\n",
    "    dataset_text_field='text', max_seq_length=1024,\n",
    "    args=TrainingArguments(\n",
    "        output_dir='/kaggle/working/stage2',\n",
    "        num_train_epochs=5,\n",
    "        per_device_train_batch_size=2,\n",
    "        gradient_accumulation_steps=4,\n",
    "        learning_rate=1e-4,\n",
    "        lr_scheduler_type='cosine',\n",
    "        warmup_ratio=0.1,\n",
    "        bf16=True, logging_steps=20,\n",
    "        eval_strategy='steps', eval_steps=100,\n",
    "        save_steps=200, save_total_limit=1,\n",
    "        report_to='none',\n",
    "    ))\n",
    "\n",
    "print('=== Stage 2 Training Start ===')\n",
    "trainer2.train()\n",
    "model2.save_pretrained('/kaggle/working/mineagent-v1-lora')\n",
    "tok2.save_pretrained('/kaggle/working/mineagent-v1-lora')\n",
    "print('Stage 2 done. Loss:', trainer2.state.log_history[-1])",
]

# Find Cell 5 (id = c6) and replace its source
for cell in nb["cells"]:
    if cell.get("id") == "c6":
        cell["source"] = NEW_CELL5_SOURCE
        print("Patched Cell 5 (id=c6)")
        break

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved:", nb_path)
print("\nNext: upload these 2 files to Kaggle dataset 'mineagent-sessions':")
print("  1. data/sessions/session_log.jsonl  (your bot sessions)")
print("  2. training/synthetic_stage2.jsonl  (1500 synthetic examples)")
