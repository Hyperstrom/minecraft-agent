import json

path = r'e:\Projects\MineCraft Agent\training\phase-3\phase3_training.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        if 'Cell 5: Advanced SFT Trainer Setup' in source:
            if 'DataCollatorForCompletionOnlyLM' not in source:
                source = source.replace('from trl import SFTTrainer\n', 'from trl import SFTTrainer, DataCollatorForCompletionOnlyLM\n')
                
                collator_code = (
                    '\n# 🛑 CRITICAL FIX: Only train on the Assistant\'s response, not the random User prompts!\n'
                    '# Qwen2.5 chat template uses <|im_start|>assistant\n'
                    'response_template = "<|im_start|>assistant\\n"\n'
                    'collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tok)\n\n'
                    'trainer = SFTTrainer('
                )
                source = source.replace('trainer = SFTTrainer(', collator_code)
                
                source = source.replace('dataset_text_field=\'text\',', 'dataset_text_field=\'text\',\n    data_collator=collator,')
                
        cell['source'] = [line + ('\n' if i < len(source.split('\n')) - 1 else '') for i, line in enumerate(source.split('\n'))]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Added DataCollatorForCompletionOnlyLM')
