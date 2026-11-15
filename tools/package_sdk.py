"""Verify installed SDKs before creating release archives."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from check_elf import inspect
from release_version import version

ROOT = Path(__file__).resolve().parents[1]


def verify_linux(prefix: Path) -> dict:
    cache = ROOT / 'build' / prefix.name / 'CMakeCache.txt'
    values = {}
    for line in cache.read_text().splitlines():
        if line.startswith(('#', '//')) or '=' not in line or ':' not in line.split('=', 1)[0]:
            continue
        key, value = line.split('=', 1)
        values[key.split(':', 1)[0]] = value
    if values.get('PHONO_DTYPE_SELECTIVE_BUILD') != 'OFF':
        raise ValueError(f'Release kernels must disable dtype pruning: {cache}')
    library = prefix / 'lib/libphono_core.so'
    report = inspect(library, max_glibc='2.28', max_glibcxx='3.4.31', max_cxxabi='1.3.14',
                     no_rpath=True, c_api_only=True)
    compiler = subprocess.check_output([values['CMAKE_CXX_COMPILER'], '-dumpfullversion'], text=True).strip()
    if compiler != '13.3.0':
        raise ValueError(f'Expected locked GCC 13.3.0, found {compiler}')
    metadata = {
        'schema_version': 1, 'c_abi': 1, 'version': version(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'compiler': compiler, 'sysroot_glibc': '2.28', 'dtype_selective_build': False,
        'sha256': hashlib.sha256(library.read_bytes()).hexdigest(), 'elf': report,
    }
    output = prefix / 'share/phono-core/build-info.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Verified {prefix.name}: {report["required_abi"]}; no RPATH; C ABI only')
    return report


def main() -> None:
    archive_format = 'zip' if os.name == 'nt' else 'gztar'
    prefixes = sorted(path for path in (ROOT / 'dist').glob('*') if path.is_dir())
    if not prefixes:
        raise ValueError('No installed SDKs found in dist; run pixi run install first')
    # Verify every input before producing any archives.
    for prefix in prefixes:
        if (prefix / 'share/phono-core/VERSION').read_text().strip() != version():
            raise ValueError(f'Installed version does not match VERSION.txt: {prefix}')
        if prefix.name.startswith('linux-'):
            verify_linux(prefix)
    output_dir = ROOT / 'artifacts'
    output_dir.mkdir(exist_ok=True)
    for prefix in prefixes:
        basename = f'phono-core-sdk-{version()}-{prefix.name}'
        shutil.make_archive(str(output_dir / basename), archive_format, prefix)


if __name__ == '__main__':
    main()
