# Build Guide

## Prerequisites

Building phono-core requires:

- pixi: creates the unified build environment (provides Python, CMake, Ninja and the C/C++ compiler, and pulls ExecuTorch's Python dependencies). Python also runs the codegen at CMake configure time, compiling the Traditional->Simplified (zh2Hans) rules into a native C++ table.
- vcpkg: provides the uni-algo and nlohmann-json C++ dependencies declared in vcpkg.json at the repository root.
- A network connection: pixi downloads Python packages and pixi run setup clones the ExecuTorch source (with submodules).

## Common Flow

From the repository root:

```
pixi run setup
pixi run config
pixi run build
```

- setup clones ExecuTorch into third_party/executorch and initializes only the submodules phono-core actually needs (the XNNPACK backend deps and third-party/flatbuffers, flatcc, json, gflags) rather than all of them; skipped/fixed-up when already present.
- config configures the CMake build, equivalent to `cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release $CMAKE_ARGS`. Extra arguments can be appended through the CMAKE_ARGS environment variable.
- build performs an incremental build, equivalent to `cmake --build build`.

When CMAKE_TOOLCHAIN_FILE is not set explicitly, CMake reads the VCPKG_ROOT environment variable and derives the vcpkg toolchain path from it, so exporting VCPKG_ROOT before configuring is all you need when using vcpkg. To use a separately installed ExecuTorch, pass -DPHONO_USE_INSTALLED_EXECUTORCH=ON and set CMAKE_PREFIX_PATH at configure time; the ExecuTorch source path is controlled by EXECUTORCH_SOURCE_DIR (default third_party/executorch).

Build artifacts are output to results/ by default:

- libphono_core.so — the C-ABI shared library
- streaming_benchmark_demo_capi — the C-ABI demo
- ime_demo_capi — the interactive C-ABI IME demo

## Selective build (operator/dtype pruning)

PhonoP2C's export.py writes ExecuTorch selective-build manifests after export
(one per model + a merged one) into PhonoP2C/export_output/manifests/. To prune
the ExecuTorch kernel library to exactly what the deployed models use, copy the
manifests into phono-core's ops_config/ directory and reconfigure:

```
cp <PhonoP2C>/export_output/manifests/<tag>_ops.yaml ops_config/
pixi run config
pixi run build
```

- Operator pruning: EXECUTORCH_SELECT_OPS_LIST registers only the operators in
  the manifests; the full portable_ops_lib is not linked.
- Dtype (precision) pruning: a selected_op_variants.h header is generated from
  the merged manifest and combined with EXECUTORCH_SELECTIVE_BUILD_DTYPE so
  portable kernels keep only the dtype variants actually used. Upstream
  ExecuTorch only supports dtype-selective-build from a single .pte model, so
  this is driven by the multi-model merged manifest instead.

Relevant CMake options:

- PHONO_OPS_CONFIG_DIR (default `ops_config/`): where manifests are read from.
- PHONO_OPS_MANIFESTS: explicit, semicolon-separated list of manifest files to
  merge (default: the `<tag>_ops.yaml` merged manifest in PHONO_OPS_CONFIG_DIR;
  if none, per-model `*_ops.yaml` files are merged).
- PHONO_DTYPE_SELECTIVE_BUILD (default ON): set OFF to keep operator pruning
  only.
- PHONO_USE_INSTALLED_EXECUTORCH, EXECUTORCH_SOURCE_DIR: unchanged.

When no manifest is present the build automatically falls back to the full
kernel library (previous behavior).

## Linux

Prerequisites: install pixi; install vcpkg and export VCPKG_ROOT; have a compatible compiler toolchain, either installed or provided by pixi.

```
# Install vcpkg (example path ~/vcpkg)
git clone https://github.com/microsoft/vcpkg ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh

# Configure and build
export VCPKG_ROOT=$HOME/vcpkg
pixi run setup
pixi run config
pixi run build
```

The pixi linux-64 platform (linux-glibc228) targets glibc 2.28, so the produced binaries run on older distributions. Keep VCPKG_ROOT exported and vcpkg will download and build uni-algo automatically; the build also requires Python3 from the pixi environment for the configure-time codegen.

## macOS

Prerequisites: install pixi and the Xcode Command Line Tools (provides clang); install vcpkg.

The pixi osx-arm64 platform targets Apple Silicon. The steps mirror Linux:

```
git clone https://github.com/microsoft/vcpkg $HOME/vcpkg
$HOME/vcpkg/bootstrap-vcpkg.sh

export VCPKG_ROOT=$HOME/vcpkg
pixi run setup
pixi run config
pixi run build
```

The dynamic library is named libphono_core.dylib on macOS, handled automatically by CMake. If the Xcode version is too old, ExecuTorch may fail to compile; upgrade Xcode and the Command Line Tools.

## Windows

Prerequisites: install pixi; install vcpkg (C:\vcpkg is a common location); install the Visual Studio 2022 C++ workload or let pixi provide the MSVC toolchain. Run from a Developer PowerShell so that CMake can locate the MSVC environment.

```
# Install vcpkg (PowerShell)
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat

# Configure and build
$env:VCPKG_ROOT="C:\vcpkg"
pixi run setup
pixi run config
pixi run build
```

On Windows, CMake defines NOMINMAX and WIN32_LEAN_AND_MEAN to avoid windows.h macro pollution, and the MSVC runtime library is fixed to the static multithreaded runtime (MT/MTd) to remove DLL dependencies. The shared library is output as phono_core.dll. When using Ninja with MSVC, run pixi run config from the Developer PowerShell or cl.exe may not be found.

## Troubleshooting

- ExecuTorch clone fails: check the network and proxy settings, delete third_party/executorch and retry pixi run setup.
- zh2Hans codegen fails: make sure pixi run config runs inside the pixi environment (the configure step invokes Python3 to run codegen/zh2hans_codegen.py); reproduce with `python codegen/zh2hans_codegen.py --input res/zh2hans.json --output src/gen/zh2hansdict.h`.
- Model fails to load: the package must use v2.1 (model_format_version is the string `"2.1"` in config.json, and the pre program exposes pre_model_pass1 and pre_model_pass2); v2 and earlier formats are unsupported.
