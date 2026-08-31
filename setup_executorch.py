import subprocess
import os
from pathlib import Path

def run(cmd, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

target_dir = Path("third_party/executorch")

# Clone if not exists
if not (target_dir / ".git").exists():
    run(f"git clone --depth 1 https://github.com/pytorch/executorch {target_dir}")

# Submodules to keep
needed = {
    "backends/xnnpack/third-party/FP16",
    "backends/xnnpack/third-party/FXdiv",
    "backends/xnnpack/third-party/XNNPACK",
    "backends/xnnpack/third-party/cpuinfo",
    "backends/xnnpack/third-party/pthreadpool",
    "third-party/flatbuffers",
    "third-party/flatcc",
    "third-party/gflags",
    "third-party/json"
}

result = subprocess.run(
    ["git", "-C", str(target_dir), "submodule", "foreach", "--quiet", "echo $name"],
    capture_output=True, text=True, check=True
)
all_subs = result.stdout.strip().split('\n')

for sub in all_subs:
    if sub and sub not in needed:
        print(f"Deinitializing {sub}...")
        run(f"git -C {target_dir} submodule deinit -f {sub} || true")

run(f"git -C {target_dir} submodule sync")
update_cmd = f"git -C {target_dir} submodule update --init --depth 1 " + " ".join(needed)
run(update_cmd)