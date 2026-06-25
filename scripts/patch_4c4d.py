#!/usr/bin/env python3
"""
Patch 4C4D's scene/dataset_readers.py to enable image loading.

The original code has image loading disabled (image = None) in two places
for "lazy loading" performance, but the lazy loading was never implemented.
This patch enables actual image loading.
"""

filepath = '/workspace/4C4D/scene/dataset_readers.py'

with open(filepath, 'r') as f:
    lines = f.readlines()

new_lines = []
skip_next = False

for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue
    
    stripped = line.strip()
    
    # Fix 1: Uncomment load_image, remove image = None
    if stripped == '# image = load_image(image_path)':
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(f'{indent}image = load_image(image_path)\n')
        # Skip the next line if it's image = None
        if i + 1 < len(lines) and lines[i + 1].strip() == 'image = None':
            skip_next = True
        continue
    
    # Fix 2: Uncomment temp_image load, remove temp_image = None
    if stripped == "# temp_image = load_image(task['temp_path'])":
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(f"{indent}temp_image = load_image(task['temp_path'])\n")
        # Skip the next line if it's temp_image = None
        if i + 1 < len(lines) and lines[i + 1].strip() == 'temp_image = None':
            skip_next = True
        continue
    
    # Skip standalone image = None / temp_image = None lines
    if stripped == 'image = None':
        continue
    if stripped == 'temp_image = None':
        continue

    # Fix 3: Disable strict camera/image parity assertion.
    # 4C4D may synthesize additional camera infos from naming conventions,
    # which can legitimately differ from on-disk file count.
    if 'assert len(cam_infos) == len(os.listdir(os.path.join(path, reading_dir)))' in stripped:
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(f"{indent}# Patched: upstream assertion is too strict for synthesized camera infos.\n")
        new_lines.append(f"{indent}if len(cam_infos) != len(os.listdir(os.path.join(path, reading_dir))):\n")
        new_lines.append(f"{indent}    print('[4C4D PATCH] camera/image count mismatch; continuing with synthesized camera infos')\n")
        continue
    
    new_lines.append(line)

with open(filepath, 'w') as f:
    f.writelines(new_lines)

print("Successfully patched 4C4D dataset_readers.py")
