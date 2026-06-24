#!/usr/bin/env python3
"""
Patch MAtCha's run_mast3r.py to handle local .pth files directly.

The original code uses from_pretrained() which goes through huggingface_hub's
repo ID validation and fails with local file paths. This patch adds a function
to load models directly from .pth files.
"""

import sys
import re

RUN_MAST3R_PATH = "/workspace/MAtCha/mast3r/run_mast3r.py"

def main():
    print(f"[PATCH] Starting MAtCha patch for {RUN_MAST3R_PATH}...")
    
    try:
        with open(RUN_MAST3R_PATH, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"[PATCH] ERROR: File not found: {RUN_MAST3R_PATH}")
        sys.exit(1)
    
    # Check if already patched
    if "load_model_from_pth" in content:
        print("[PATCH] Already patched, skipping.")
        return
    
    # Find the line with model = AsymmetricMASt3R.from_pretrained
    pattern = r'^(\s+)model\s*=\s*AsymmetricMASt3R\.from_pretrained\(args\.weights_path\)\.to\(device\)'
    match = re.search(pattern, content, re.MULTILINE)
    
    if not match:
        print("[PATCH] ERROR: Could not find model loading line")
        print("[PATCH] Searching for similar patterns...")
        for line_num, line in enumerate(content.split('\n'), 1):
            if 'AsymmetricMASt3R' in line and 'from_pretrained' in line:
                print(f"[PATCH] Found at line {line_num}: {line.strip()}")
        sys.exit(1)

    indent = match.group(1)
    print(f"[PATCH] Found line with indent: {repr(indent)}")

    # Create the replacement
    replacement = f'''{indent}# Handle local .pth files directly
{indent}if args.weights_path.endswith(".pth") and os.path.exists(args.weights_path):
{indent}    import torch
{indent}    from mast3r.model import AsymmetricMASt3R
{indent}    ckpt = torch.load(args.weights_path, map_location="cpu")
{indent}    if "args" in ckpt and "model" in ckpt:
{indent}        model_args = vars(ckpt["args"]) if hasattr(ckpt["args"], '__dict__') else ckpt["args"]
{indent}        # Filter out non-constructor arguments
{indent}        valid_args = {{k: v for k, v in model_args.items() if k not in ['model', 'state_dict', 'optimizer']}}
{indent}        # Fix img_size if it's an int instead of tuple
{indent}        if 'img_size' in valid_args and isinstance(valid_args['img_size'], int):
{indent}            valid_args['img_size'] = (valid_args['img_size'], valid_args['img_size'])
{indent}        model = AsymmetricMASt3R(**valid_args)
{indent}        model.load_state_dict(ckpt["model"])
{indent}    elif "state_dict" in ckpt:
{indent}        model = AsymmetricMASt3R()
{indent}        model.load_state_dict(ckpt["state_dict"])
{indent}    else:
{indent}        model = AsymmetricMASt3R()
{indent}        model.load_state_dict(ckpt)
{indent}    model = model.to(device)
{indent}else:
{indent}    model = AsymmetricMASt3R.from_pretrained(args.weights_path).to(device)'''
    
    # Replace the line
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Make sure 'import os' is present
    if 'import os' not in new_content:
        print("[PATCH] Adding 'import os'...")
        lines = new_content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import '):
                lines.insert(i + 1, 'import os')
                break
        new_content = '\n'.join(lines)
    
    # Write back
    with open(RUN_MAST3R_PATH, 'w') as f:
        f.write(new_content)
    
    print("[PATCH] Successfully patched run_mast3r.py")
    print(
        f"[PATCH] Verification: 'load_model_from_pth' in file: {'load_model_from_pth' in new_content}")

if __name__ == "__main__":
    main()
