#!/usr/bin/env python3
"""Configure platform presets without leaking Conda's development RPATH into releases."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def release_environment(environment: dict[str, str], build: Path, preset: str) -> dict[str, str]:
    environment = environment.copy()
    if preset.startswith('android-'):
        ndk = Path(environment.get('ANDROID_NDK_ROOT') or
                   environment.get('ANDROID_NDK_HOME', '')).resolve()
        if not (ndk / 'build/cmake/android.toolchain.cmake').is_file():
            raise ValueError('Set ANDROID_NDK_ROOT to the installed Android NDK')
        environment['ANDROID_NDK_ROOT'] = str(ndk)
        environment['ANDROID_NDK_HOME'] = str(ndk)
        # The NDK owns target flags and library searches. In particular, a host
        # -L<conda>/lib makes Android's -latomic select the Linux libatomic.so.
        # Keep CC/CXX available to ExecuTorch's separate host-tool sub-builds.
        for variable in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS'):
            environment[variable] = ''
        return environment
    # These variables select Visual Studio's architecture/toolset, not Ninja's.
    if os.name == 'nt':
        environment.pop('CMAKE_GENERATOR_PLATFORM', None)
        environment.pop('CMAKE_GENERATOR_TOOLSET', None)
        return environment
    flags = shlex.split(environment.get('LDFLAGS', ''))
    flags = [flag for flag in flags if not flag.startswith(('-Wl,-rpath,', '-Wl,-rpath='))]
    if preset.startswith('linux-'):
        # Compiler specs can add an RPATH even after LDFLAGS is cleaned.
        for variable in ('CC', 'CXX'):
            compiler = environment[variable]
            version = subprocess.check_output([compiler, '-dumpfullversion'], text=True).strip()
            if version != '13.3.0':
                raise ValueError(f'{variable}: expected locked GCC 13.3.0, found {version}')
        sysroot = Path(environment['CONDA_BUILD_SYSROOT']).resolve(strict=True)
        prefix = Path(environment['CONDA_PREFIX']).resolve(strict=True)
        if not sysroot.is_relative_to(prefix) or not any(sysroot.glob('lib*/libc-2.28.so')):
            raise ValueError(f'Expected the locked glibc 2.28 sysroot, found {sysroot}')
        actual_sysroot = Path(subprocess.check_output(
            [environment['CXX'], '-print-sysroot'], text=True).strip()).resolve(strict=True)
        if actual_sysroot != sysroot:
            raise ValueError(f'Compiler sysroot {actual_sysroot} differs from {sysroot}')
        specs = subprocess.check_output([environment['CXX'], '-dumpspecs'], text=True)
        specs = re.sub(r'%\{!static:-rpath [^}\n]+\}', '', specs)
        build.mkdir(parents=True, exist_ok=True)
        release_specs = build / 'release.specs'
        release_specs.write_text(specs)
        flags.append(f'-specs={release_specs}')
    environment['LDFLAGS'] = shlex.join(flags)
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', default=os.environ.get('CMAKE_BUILD_DIR'))
    args, extra = parser.parse_known_args()
    preset = args.preset
    if not preset:
        parser.error('Run through Pixi or specify --preset')
    environment = release_environment(os.environ.copy(), ROOT / 'build' / preset, preset)
    arguments = ['cmake', '--preset', preset]
    if os.name != 'nt':
        # Explicit cache values also replace contaminated flags on reconfigure.
        for kind in ('EXE', 'SHARED', 'MODULE'):
            arguments.append(f'-DCMAKE_{kind}_LINKER_FLAGS={environment["LDFLAGS"]}')
    subprocess.run([*arguments, *extra], env=environment, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
