import json

path = r'e:\Projects\MineCraft Agent\training\phase-3\phase3_training.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Optimize Batch Size & Epochs
        if 'Cell 5: Advanced SFT Trainer Setup' in source:
            source = source.replace('num_train_epochs=15,             # High epochs for strict JSON memorization', 'num_train_epochs=5,              # 5 epochs per run (under 10 hours)')
            source = source.replace('per_device_train_batch_size=2,   # 🛑 CRITICAL: Keep at 2 to avoid VRAM limits', 'per_device_train_batch_size=16,  # 🚀 Maximize GPU Utilization (Uses ~10GB VRAM)')
            source = source.replace('per_device_train_batch_size=4,', 'per_device_train_batch_size=16,')
            source = source.replace('gradient_accumulation_steps=32,  # Effective Batch Size = 2 * 32 = 64', 'gradient_accumulation_steps=2,   # Effective Batch Size = 32')
            
        cell['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

# Add a new Markdown cell at the end explaining how to resume
resume_cell = {
 'id': 'p3_resume',
 'cell_type': 'markdown',
 'source': [
  '## 🔄 How to Train the Next 5 Epochs (Part 2)\n',
  'Because Kaggle limits sessions to 12 hours, we are doing **5 epochs per notebook**.\n',
  'To train the next 5 epochs, simply:\n',
  '1. Upload the saved `/kaggle/working/mineagent-phase3-lora` folder as a new Kaggle Dataset (e.g. `mineagent-part1`).\n',
  '2. In your Part 2 notebook, change `MODEL_ID` in Cell 3 to the path of your dataset:\n',
  '   `MODEL_ID = \'/kaggle/input/mineagent-part1/mineagent-phase3-lora\'`\n',
  '3. Run the notebook again! It will stack the new training on top of your previous progress.'
 ],
 'metadata': {}
}

if not any('How to Train the Next 5 Epochs' in ''.join(c['source']) for c in nb['cells']):
    nb['cells'].insert(-2, resume_cell)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Optimized GPU utilization and added 5-epoch split strategy')
