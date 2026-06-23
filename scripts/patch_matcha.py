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

# The function to inject
LOAD_MODEL_FUNC = '''
def load_model_from_pth(path, device):
    """Load AsymmetricMASt3R model directly from a .pth checkpoint file."""
    import torch
    from mast3r.model import AsymmetricMASt3R
    ckpt = torch.load(path, map_location="cpu")
    # Handle different checkpoint formats
    if "args" in ckpt and "model" in ckpt:
        model = AsymmetricMASt3R(**ckpt["args"])
        model.load_state_dict(ckpt["model"])
    elif "state_dict" in ckpt:
        model = AsymmetricMASt3R()
        model.load_state_dict(ckpt["state_dict"])
    else:
        # Try loading as state dict directly
        model = AsymmetricMASt3R()
        model.load_state_dict(ckpt)
    return model.to(device)

'''

def main():
    print(f"Patching {RUN_MAST3R_PATH}...")
    
    with open(RUN_MAST3R_PATH, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "load_model_from_pth" in content:
        print("Already patched, skipping.")
        return
    
    # Find the line with model = AsymmetricMASt3R.from_pretrained
    pattern = r'^(\s+)model\s*=\s*AsymmetricMASt3R\.from_pretrained\(args\.weights_path\)\.to\(device\)'
    match = re.search(pattern, content, re.MULTILINE)
    
    if not match:
        print("ERROR: Could not find model loading line")
        sys.exit(1)

    indent = match.group(1)
    print(f"Found line with indent: {repr(indent)}")

    # Create replacement code
    replacement = f'''{indent}# Handle local .pth files directly
{indent}if args.weights_path.endswith(".pth") and os.path.exists(args.weights_path):
{indent}    model = load_model_from_pth(args.weights_path, device)
{indent}else:
{indent}    model = AsymmetricMASt3R.from_pretrained(args.weights_path).to(device)'''
    
    # Replace the line
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Insert the helper function before the if __name__ block
    if "__name__" in content:
        # Find the line with if __name__ == "__main__":
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '__name__' in line and '__main__' in line:
                # Insert before this line
                lines.insert(i, LOAD_MODEL_FUNC)
                break
        content = '\n'.join(lines)
    
    # Make sure 'import os' is present
    if 'import os' not in content:
        # Add it after the first import
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import '):
                lines.insert(i + 1, 'import os')
                break
        content = '\n'.join(lines)
    
    with open(RUN_MAST3R_PATH, 'w') as f:
        f.write(content)
    
    print("Successfully patched run_mast3r.py")

if __name__ == "__main__":
    main()
