#!/usr/bin/env python3
"""
Patch 4C4D's train.py to fix PyTorch 2.0.1 compatibility.

PyTorch 2.0.1 doesn't support the 'expandable_segments' option in
PYTORCH_CUDA_ALLOC_CONF. This patch removes or comments out that setting.
"""

import sys

TRAIN_PY_PATH = "/workspace/4C4D/train.py"

def main():
    print(f"[PATCH] Checking {TRAIN_PY_PATH}...")
    
    try:
        with open(TRAIN_PY_PATH, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[PATCH] ERROR: File not found: {TRAIN_PY_PATH}")
        sys.exit(1)
    
    # Find and comment out the expandable_segments line
    new_lines = []
    patched = False
    for i, line in enumerate(lines):
        if 'PYTORCH_CUDA_ALLOC_CONF' in line and 'expandable_segments' in line:
            # Comment out this line
            new_lines.append(f"# {line}")
            patched = True
            print(f"[PATCH] Commented out line {i+1}: {line.strip()}")
        else:
            new_lines.append(line)
    
    if patched:
        with open(TRAIN_PY_PATH, 'w') as f:
            f.writelines(new_lines)
        print("[PATCH] Successfully patched train.py")
    else:
        print("[PATCH] No expandable_segments setting found - no patch needed")

if __name__ == "__main__":
    main()
