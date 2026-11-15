"""Verify installed SDKs before creating release archives."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from check_elf import inspect
from check_macho import inspect as inspect_macho
from release_version import version

ROOT = Path(__file__).resolve().parents[1]
# GCC 13.2 introduced GLIBCXX_3.4.32; the pinned 13.3 toolchains can use it.
# https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html
LINUX_ABI_LIMITS = dict(max_glibc='2.28', max_glibcxx='3.4.32', max_cxxabi='1.3.14')


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
    report = inspect(library, **LINUX_ABI_LIMITS, no_rpath=True, c_api_only=True)
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


def verify_macos(prefix: Path) -> dict:
    library = prefix / 'lib/libphono_core.dylib'
    report = inspect_macho(library)
    cache = ROOT / 'build' / prefix.name / 'CMakeCache.txt'
    values = {line.split('=', 1)[0].split(':', 1)[0]: line.split('=', 1)[1]
              for line in cache.read_text().splitlines()
              if not line.startswith(('#', '//')) and '=' in line}
    if values.get('PHONO_DTYPE_SELECTIVE_BUILD') != 'OFF':
        raise ValueError('Release kernels must disable dtype pruning')
    compiler = values['CMAKE_CXX_COMPILER']
    compiler_version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    if not compiler_version.startswith('Apple clang version '):
        raise ValueError(f'Expected Xcode Apple Clang, found {compiler_version}')
    metadata = {
        'schema_version': 1, 'c_abi': 1, 'version': version(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'compiler': compiler_version, 'dtype_selective_build': False,
        'sha256': hashlib.sha256(library.read_bytes()).hexdigest(), 'macho': report,
    }
    (prefix / 'share/phono-core/build-info.json').write_text(json.dumps(metadata, indent=2) + '\n')
    # Relocate the installed SDK and consume its CMake target from pure C.
    # No Pixi/compiler activation or DYLD override may satisfy its imports.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('DYLD_', 'CONDA_')) and key not in
                   {'CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'SDKROOT',
                    'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH',
                    'CMAKE_PREFIX_PATH', 'CMAKE_TOOLCHAIN_FILE'}}
    environment['PATH'] = '/usr/bin:/bin:/usr/sbin:/sbin'
    cmake = shutil.which('cmake')
    with tempfile.TemporaryDirectory(dir=ROOT / 'build') as temporary:
        root = Path(temporary)
        sdk = root / 'relocated-sdk'
        shutil.copytree(prefix, sdk)
        source = root / 'consumer'
        source.mkdir()
        (source / 'main.c').write_text(
            '#include <phono_api.h>\n'
            'int main(void) { return phono_engine_create(0, 0, 0) == PHONO_INVALID_ARGUMENT ? 0 : 1; }\n')
        (source / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.24)\nproject(consumer LANGUAGES C)\n'
            'find_package(phono-core CONFIG REQUIRED)\nadd_executable(consumer main.c)\n'
            'target_link_libraries(consumer PRIVATE phono::core)\n')
        subprocess.run([cmake, '-S', source, '-B', root / 'consumer-build', '-G', 'Unix Makefiles',
                        '-DCMAKE_C_COMPILER=/usr/bin/clang', f'-DCMAKE_PREFIX_PATH={sdk}',
                        '-DCMAKE_OSX_ARCHITECTURES=arm64', '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0'],
                       env=environment, check=True)
        subprocess.run([cmake, '--build', root / 'consumer-build'], env=environment, check=True)
        subprocess.run([root / 'consumer-build/consumer'], env=environment, check=True)
    print(f'Verified {prefix.name}: system libc++; no RPATH; C ABI only; relocated C consumer loaded')
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
        elif prefix.name.startswith('macos-'):
            verify_macos(prefix)
    output_dir = ROOT / 'artifacts'
    output_dir.mkdir(exist_ok=True)
    for prefix in prefixes:
        basename = f'phono-core-sdk-{version()}-{prefix.name}'
        shutil.make_archive(str(output_dir / basename), archive_format, prefix)


if __name__ == '__main__':
    main()
