#!/usr/bin/env python3
"""
Patch MAtCha's run_mast3r.py to handle local .pth files directly.

The original code uses from_pretrained() which goes through huggingface_hub's
repo ID validation and fails with local file paths. This patch adds a function
to load models directly from .pth files.
"""

import sys

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

# The replacement for the model loading line
OLD_LINE = "    model = AsymmetricMASt3R.from_pretrained(args.weights_path).to(device)"
NEW_CODE = """    # Handle local .pth files directly
    if args.weights_path.endswith(".pth") and os.path.exists(args.weights_path):
        model = load_model_from_pth(args.weights_path, device)
    else:
        model = AsymmetricMASt3R.from_pretrained(args.weights_path).to(device)"""

def main():
    print(f"Patching {RUN_MAST3R_PATH}...")
    
    with open(RUN_MAST3R_PATH, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "load_model_from_pth" in content:
        print("Already patched, skipping.")
        return
    
    # Find where to insert the function (before the line that loads the model)
    lines = content.split('\n')
    insert_idx = None
    
    for i, line in enumerate(lines):
        if OLD_LINE in line:
            # Find the start of the function containing this line
            for j in range(i-1, -1, -1):
                if lines[j].startswith('def '):
                    insert_idx = j
                    break
            break
    
    if insert_idx is None:
        print(f"ERROR: Could not find line to patch: {OLD_LINE}")
        sys.exit(1)
    
    # Insert the function
    lines.insert(insert_idx, LOAD_MODEL_FUNC)
    
    # Now find and replace the model loading line
    content = '\n'.join(lines)
    content = content.replace(OLD_LINE, NEW_CODE)
    
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
