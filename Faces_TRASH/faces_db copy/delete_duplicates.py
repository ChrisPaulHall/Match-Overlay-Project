import os
import hashlib

# CONFIG: Set your image folder path
faces_db = "faces_db"  # Update if needed

# Helper function to generate MD5 hash from a file
def hash_file(filepath):
    with open(filepath, 'rb') as f:
        hash_md5 = hashlib.md5()
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
        return hash_md5.hexdigest()

# Scan folder and group files by content hash
hash_to_files = {}
for filename in os.listdir(faces_db):
    if filename.lower().endswith(".jpg"):
        path = os.path.join(faces_db, filename)
        file_hash = hash_file(path)
        if file_hash in hash_to_files:
            hash_to_files[file_hash].append(path)
        else:
            hash_to_files[file_hash] = [path]

# Identify and delete duplicates (keep the first file only)
deleted = []
for files in hash_to_files.values():
    if len(files) > 1:
        for duplicate_path in files[1:]:
            os.remove(duplicate_path)
            deleted.append(os.path.basename(duplicate_path))

# Report results
print(f"✅ Duplicate removal complete. {len(deleted)} files deleted:")
for f in deleted:
    print("  🗑️", f)
