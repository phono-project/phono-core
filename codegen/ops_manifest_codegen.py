"""Merge one or more ExecuTorch selective-build manifests into a single one.

PhonoP2C/export.py emits per-model operator manifests (and a merged one) in
the ExecuTorch ``selected_operators.yaml`` format. phono-core consumes them to
prune the ExecuTorch kernel library down to just the operators (and, when
combined with dtype-selective build, the dtypes) the deployed models use.

This script merges the input manifests (union of ``operators`` and of
``et_kernel_metadata`` per operator) and emits:

  * ``--output``              the merged manifest (selected_operators.yaml)
  * ``--root-ops-output``     a comma-separated operator list for ExecuTorch's
                              EXECUTORCH_SELECT_OPS_LIST option
"""

import argparse
import sys
from typing import Any, Dict, List

import yaml


def _merge(manifests: List[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "build_features": [],
        "custom_classes": [],
        "debug_info": [],
        "et_kernel_metadata": {},
        "include_all_non_op_selectives": False,
        "include_all_operators": False,
        "kernel_metadata": {},
        "operators": {},
    }
    for path in manifests:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        merged["include_all_operators"] = merged["include_all_operators"] or bool(
            data.get("include_all_operators", False)
        )
        for op_name, op_info in (data.get("operators") or {}).items():
            if op_name not in merged["operators"]:
                merged["operators"][op_name] = op_info
        for op_name, kernel_keys in (data.get("et_kernel_metadata") or {}).items():
            seen = merged["et_kernel_metadata"].setdefault(op_name, [])
            for key in kernel_keys or []:
                if key not in seen:
                    seen.append(key)
    return merged


def _dump(data: Dict[str, Any], path: str) -> None:
    with open(path, "wb") as f:
        f.write(yaml.safe_dump(data, default_flow_style=False).encode("utf-8"))


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--output",
        required=True,
        help="merged selected_operators.yaml",
    )
    parser.add_argument(
        "--root-ops-output",
        required=True,
        help="comma-separated operator list for EXECUTORCH_SELECT_OPS_LIST",
    )
    parser.add_argument(
        "manifest",
        nargs="+",
        help="input manifest yaml files",
    )
    options = parser.parse_args(argv)

    if not options.manifest:
        print(
            "ops_manifest_codegen: no manifest inputs given",
            file=sys.stderr,
        )
        return 2

    merged = _merge(options.manifest)
    _dump(merged, options.output)

    if merged["include_all_operators"]:
        root_ops = ""
    else:
        root_ops = ",".join(sorted(merged["operators"].keys()))
    with open(options.root_ops_output, "w", encoding="utf-8") as f:
        f.write(root_ops)

    print(
        f"ops_manifest_codegen: merged {len(options.manifest)} manifest(s) -> "
        f"{len(merged['operators'])} operators"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
