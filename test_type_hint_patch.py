
import sys
import os

sys.path.append(os.getcwd())

try:
    from nodes.nodes_training import VibeVoice_LoRA_Trainer
    trainer = VibeVoice_LoRA_Trainer()

    repo_dir = 'vibevoice_finetuning_repo'
    patience = 42
    threshold = 0.007
    save_total_limit = 5

    # Run the patch
    success = trainer._patch_early_stopping(repo_dir, patience, threshold, save_total_limit)
    print(f'Patch returned: {success}')

    with open('vibevoice_finetuning_repo/src/finetune_vibevoice_lora.py', 'r') as f:
        content = f.read()

        if 'class SmartEarlyStoppingAndSaveCallback(TrainerCallback):' in content:
             print('PASSED: New callback class injected')
        else:
             print('FAILED: New callback class NOT injected')

        # Verify it was injected before main() -> None:
        idx_class = content.find('class SmartEarlyStoppingAndSaveCallback')
        idx_main = content.find('def main() -> None:')

        if idx_class != -1 and idx_main != -1 and idx_class < idx_main:
             print('PASSED: Callback injected before main()')
        else:
             print(f'FAILED: Injection order incorrect. Class idx: {idx_class}, Main idx: {idx_main}')

except Exception as e:
    print(f'Exception: {e}')
