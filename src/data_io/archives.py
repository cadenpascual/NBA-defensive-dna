from pathlib import Path
import json
import shutil
from py7zr import SevenZipFile

def extract_and_load_json(archive_path, tmp_root="data/tmp/json"):
    """
    Extract a .7z archive containing a single JSON file,
    load it safely, then clean up.
    """
    archive_path = Path(archive_path)
    tmp_dir = Path(tmp_root) / archive_path.stem
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=tmp_dir)

    json_files = list(tmp_dir.glob("*.json"))
    if len(json_files) != 1:
        shutil.rmtree(tmp_dir)
        print(f"Warning: Expected 1 JSON in {archive_path}, found {len(json_files)}")
        return None

    json_path = json_files[0]

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Corrupt JSON in {archive_path}")
        shutil.rmtree(tmp_dir)
        return None

    shutil.rmtree(tmp_dir)  # clean extraction dir
    return data
