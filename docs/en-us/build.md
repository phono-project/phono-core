# Build Guide

## Prerequisites

Building phono-core requires:

- pixi: creates the unified build environment (provides Python, CMake, Ninja and the C/C++ compiler, and pulls ExecuTorch's Python dependencies).
- vcpkg: provides the ICU and nlohmann-json C++ dependencies declared in vcpkg.json at the repository root. If ICU is already installed system-wide, CMake can use it directly without vcpkg.
- A network connection: pixi downloads Python packages and pixi run setup clones the ExecuTorch source (with submodules).

## Common Flow

From the repository root:

```
pixi run setup
pixi run config
pixi run build
```

- setup clones ExecuTorch (with submodules) into third_party/executorch; skipped if already present.
- config configures the CMake build, equivalent to `cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release $CMAKE_ARGS`. Extra arguments can be appended through the CMAKE_ARGS environment variable.
- build performs an incremental build, equivalent to `cmake --build build`.

When CMAKE_TOOLCHAIN_FILE is not set explicitly, CMake reads the VCPKG_ROOT environment variable and derives the vcpkg toolchain path from it, so exporting VCPKG_ROOT before configuring is all you need when using vcpkg. To use a separately installed ExecuTorch, pass -DPHONO_USE_INSTALLED_EXECUTORCH=ON and set CMAKE_PREFIX_PATH at configure time; the ExecuTorch source path is controlled by EXECUTORCH_SOURCE_DIR (default third_party/executorch).

Build artifacts are output to results/ by default:

- libphono_core.so — the C-ABI shared library
- streaming_benchmark_demo — the C++ API demo
- streaming_benchmark_demo_capi — the C-ABI demo

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

The pixi linux-64 platform (linux-glibc228) targets glibc 2.28, so the produced binaries run on older distributions. If ICU is missing system-wide, keep VCPKG_ROOT exported and vcpkg will download and build ICU automatically; if system ICU is available and you want to skip vcpkg, override the toolchain file at configure time and let CMake find the system ICU.

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
- vcpkg builds ICU slowly: the first configure downloads and compiles ICU, which is expected to take a while; run pixi run config early to warm it up.
- ICU not found: make sure VCPKG_ROOT is exported, or install ICU system-wide (libicu-dev on Linux, icu4c via brew on macOS).
- Model fails to load: the package must be v2 format (model_format_version is 2 in config.json, and the pre program exposes pre_model_pass1 and pre_model_pass2).
