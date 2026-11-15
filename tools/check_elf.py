"""Inspect actual ELF imports, exported ABI, loader paths and runtime dependencies."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


CORE_RUNTIME = {
    "libc.so.6", "libm.so.6", "libstdc++.so.6", "libgcc_s.so.1",
    "libpthread.so.0", "libdl.so.2", "librt.so.1", "ld-linux-x86-64.so.2",
    "ld-linux-aarch64.so.1",
}


def readelf(path: Path, option: str, program: str) -> str:
    return subprocess.check_output(
        [program, option, "--wide", str(path)], text=True,
        env={**os.environ, "LC_ALL": "C"},
    )


def number(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def inspect(path: Path, *, max_glibc: str | None = None,
            max_glibcxx: str | None = None, max_cxxabi: str | None = None,
            no_rpath: bool = False, c_api_only: bool = False,
            program: str = "readelf") -> dict:
    path = path.resolve(strict=True)
    header = readelf(path, "--file-header", program)
    if not re.search(r"Type:\s+DYN\b", header):
        raise ValueError(f"Expected an ELF shared library: {path}")
    versions = readelf(path, "--version-info", program)
    # Version definitions identify this library's exports; compatibility is
    # determined by version *requirements* from its imported symbols.
    required_section = versions.split("Version needs section", 1)
    names = re.findall(r"\bName:\s+(\S+)", required_section[1]) if len(required_section) > 1 else []
    maxima = {}
    for family, limit in (("GLIBC", max_glibc), ("GLIBCXX", max_glibcxx), ("CXXABI", max_cxxabi)):
        values = []
        for name in names:
            if not name.startswith(family + "_"): continue
            value = name[len(family) + 1:]
            if not re.fullmatch(r"\d+(?:\.\d+)+", value):
                raise ValueError(f"Unexpected private/unversioned runtime ABI {name} in {path}")
            values.append(value)
        maximum = max(values, key=number) if values else None
        maxima[family] = maximum
        if maximum and limit and number(maximum) > number(limit):
            raise ValueError(f"{path.name} requires {family}_{maximum}, above allowed {family}_{limit}")
    dynamic = readelf(path, "--dynamic", program)
    needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic)
    rpaths = re.findall(r"\((?:RPATH|RUNPATH)\).*?\[(.*?)\]", dynamic)
    if no_rpath and rpaths:
        raise ValueError(f"Loader paths are forbidden in {path.name}: {rpaths}")
    exports = []
    symbols = readelf(path, "--dyn-syms", program)
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].endswith(":"): continue
        # num, value, size, type, bind, visibility, section, symbol name
        if fields[4] not in ("GLOBAL", "WEAK") or fields[5] != "DEFAULT": continue
        if fields[6] in ("UND", "ABS"): continue
        exports.append(fields[7].split("@", 1)[0])
    exports = sorted(set(exports))
    if c_api_only:
        if not exports or "phono_engine_create" not in exports:
            raise ValueError(f"Missing Phono C API exports in {path}")
        unexpected = [name for name in exports if not name.startswith("phono_")]
        if unexpected:
            raise ValueError(f"Non-C-API symbols exported by {path.name}: {unexpected[:10]}")
        unexpected_runtime = sorted(set(needed) - CORE_RUNTIME)
        if unexpected_runtime:
            raise ValueError(f"Unexpected runtime dependencies in {path.name}: {unexpected_runtime}")
    return {"path": str(path), "required_abi": maxima, "needed": needed,
            "loader_paths": rpaths, "export_count": len(exports), "exports": exports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", nargs="+", type=Path)
    parser.add_argument("--max-glibc")
    parser.add_argument("--max-glibcxx")
    parser.add_argument("--max-cxxabi")
    parser.add_argument("--no-rpath", action="store_true")
    parser.add_argument("--c-api-only", action="store_true")
    parser.add_argument("--readelf", default=os.environ.get("READELF", "readelf"))
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()
    try:
        reports = [inspect(path, max_glibc=args.max_glibc, max_glibcxx=args.max_glibcxx,
                           max_cxxabi=args.max_cxxabi, no_rpath=args.no_rpath,
                           c_api_only=args.c_api_only, program=args.readelf) for path in args.elf]
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error
    result = {"schema_version": 1, "libraries": reports}
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(result, indent=2) + "\n")
    for report in reports:
        print(f"{Path(report['path']).name}: ABI={report['required_abi']}, "
              f"NEEDED={report['needed']}, exports={report['export_count']}, "
              f"loader paths={report['loader_paths']}")


if __name__ == "__main__":
    main()
