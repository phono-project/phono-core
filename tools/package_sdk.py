from pathlib import Path
import shutil
import os

from release_version import version

ROOT = Path(__file__).resolve().parents[1]

format = "zip" if os.name == "nt" else "gztar"

for prefix in ROOT.glob("dist/*"):
    if prefix.is_file():
        continue
    output_dir = ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)
    basename = f"phono-core-sdk-{version()}-{prefix.name}"
    shutil.make_archive(str(output_dir / basename), format, prefix)
