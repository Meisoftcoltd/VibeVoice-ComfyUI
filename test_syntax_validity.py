
import sys
import os
import ast

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

        # Check for escaped newlines (which were causing SyntaxError)
        if '\\n' in content:
             print('FAILED: Found double escaped newlines (potential SyntaxError source)')
        else:
             print('PASSED: No double escaped newlines found')

        # Verify syntax validity
        try:
            ast.parse(content)
            print('PASSED: Patched code has valid Python syntax')
        except SyntaxError as e:
            print(f'FAILED: Patched code has SyntaxError: {e}')
            print('--- Error Context ---')
            lines = content.splitlines()
            if e.lineno:
                start = max(0, e.lineno - 5)
                end = min(len(lines), e.lineno + 5)
                for i in range(start, end):
                    print(f'{i+1}: {lines[i]}')

        # Verify print statements are cleaned
        if 'print("")' in content:
             print('PASSED: Found empty print statements for formatting')
        else:
             print('FAILED: Did not find expected empty print statements')

except Exception as e:
    print(f'Exception: {e}')
