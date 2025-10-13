import os

root = "data/processed"

for subdir, dirs, files in os.walk(root):
    print(f"{subdir} -> {len(files)} files")
