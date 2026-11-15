"""Audit the installed macOS C ABI library before SDK publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

SYSTEM_LIBRARIES = {'/usr/lib/libSystem.B.dylib', '/usr/lib/libc++.1.dylib'}


def inspect(path: Path) -> dict:
    architectures = subprocess.check_output(['/usr/bin/lipo', '-archs', path], text=True).strip().split()
    if architectures != ['arm64']:
        raise ValueError(f'Expected an ARM64 Mach-O library: {architectures}')
    commands = subprocess.check_output(['/usr/bin/otool', '-l', path], text=True)
    needed, rpaths, identities, minima = [], [], [], []
    for command in re.split(r'Load command \d+\n', commands)[1:]:
        kind = re.search(r'^\s*cmd (\S+)', command, re.M).group(1)
        if kind == 'LC_RPATH':
            rpaths.append(re.search(r'\bpath (.*?) \(offset', command).group(1))
        elif kind in {'LC_LOAD_DYLIB', 'LC_LOAD_WEAK_DYLIB', 'LC_REEXPORT_DYLIB', 'LC_LOAD_UPWARD_DYLIB', 'LC_ID_DYLIB'}:
            name = re.search(r'\bname (.*?) \(offset', command).group(1)
            (identities if kind == 'LC_ID_DYLIB' else needed).append(name)
        elif kind in {'LC_BUILD_VERSION', 'LC_VERSION_MIN_MACOSX'}:
            minima.append(re.search(r'^\s*(?:minos|version) ([0-9.]+)', command, re.M).group(1))
    if identities != ['@rpath/libphono_core.dylib']:
        raise ValueError(f'Non-relocatable Mach-O install name: {identities}')
    if rpaths:
        raise ValueError(f'Loader paths are forbidden in the core dylib: {rpaths}')
    unexpected = set(needed) - SYSTEM_LIBRARIES
    if unexpected or not SYSTEM_LIBRARIES.issubset(needed):
        raise ValueError(f'Core must use system libc++ and libSystem only: {needed}')
    if len(minima) != 1 or tuple(map(int, minima[0].split('.'))) > (14, 0, 0):
        raise ValueError(f'Expected macOS deployment target no newer than 14.0: {minima}')
    symbols = subprocess.check_output(['/usr/bin/nm', '-gUj', path], text=True).splitlines()
    exports = sorted(symbol.removeprefix('_') for symbol in symbols)
    if 'phono_engine_create' not in exports or any(not symbol.startswith('phono_') for symbol in exports):
        raise ValueError(f'Expected only Phono C ABI exports: {exports[:10]}')
    return dict(architectures=architectures, install_name=identities[0], minimum_macos=minima[0],
                needed=needed, loader_paths=rpaths, export_count=len(exports), exports=exports)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library', type=Path)
    print(json.dumps(inspect(parser.parse_args().library.resolve(strict=True)), indent=2))
