import os
import re

faces_dir = os.path.dirname(os.path.abspath(__file__))
image_files = [f for f in os.listdir(faces_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Group files by base name (e.g., ABera)
groups = {}
for fname in image_files:
    match = re.match(r"^([A-Za-z]+)(\d*)\.(jpg|jpeg|png)$", fname)
    if match:
        base = match.group(1)
        groups.setdefault(base, []).append(fname)

for base, files in groups.items():
    # Sort files by any number in their name
    files_sorted = sorted(files, key=lambda x: int(re.search(r"(\d+)", x).group(1)) if re.search(r"(\d+)", x) else 0)
    for idx, fname in enumerate(files_sorted):
        ext = fname.split('.')[-1]
        # First file: ABera.jpg, others: ABera1.jpg, ABera2.jpg, ...
        new_name = f"{base}.jpg" if idx == 0 else f"{base}{idx}.jpg"
        old_path = os.path.join(faces_dir, fname)
        new_path = os.path.join(faces_dir, new_name)
        if old_path != new_path:
            print(f"Renaming {fname} -> {new_name}")
            os.rename(old_path, new_path)