import rarfile
import os

rar_path = r"data/raw/DatabaseSubjects.rar"
extract_path = r"data/raw/DatabaseSubjects"

os.makedirs(extract_path, exist_ok=True)

with rarfile.RarFile(rar_path) as rf:
    rf.extractall(extract_path)

print("Extraction complete ✅")