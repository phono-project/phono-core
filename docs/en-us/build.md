# Build guide

[Project home](../../README.md) · [中文](../zh-cn/build.md)

## Environment and build entry points

Install Pixi and a C++ toolchain. On Windows, use a Visual Studio x64 developer shell with the C++ Desktop workload and Windows SDK installed, and set `VCPKG_ROOT`. Linux uses the locked GCC 13.3.0 and glibc 2.28 sysroot in the same Pixi environment. See [SDK, compatibility and releases](../sdk-release.md) for standard SDK builds, platform requirements, and release validation.

For routine source builds in a prepared development environment:

```sh
pixi install --locked
pixi run setup
pixi run config
pixi run build
pixi run test
```

`setup` prepares the ExecuTorch source and required submodules according to `justfile`. `config` configures CMake, and `build` builds incrementally, with executables under `build/<preset>/bin/` by default. CMake obtains uni-algo with FetchContent and uses the ExecuTorch JSON headers; `codegen/zh2hans_codegen.py` generates C++ tables for traditional-to-simplified conversion during configuration.

## Build options

The following options can be set during CMake configuration:

| Option | Purpose |
| --- | --- |
| `PHONO_USE_INSTALLED_EXECUTORCH` | Use a separately installed ExecuTorch; also set `CMAKE_PREFIX_PATH` |
| `EXECUTORCH_SOURCE_DIR` | ExecuTorch source path; default: `third_party/executorch` |
| `PHONO_XNNPACK_WEIGHT_CACHE` | Share XNNPACK packed weights; default: ON |
| `PHONO_OPS_CONFIG_DIR` | Operator manifest directory; default: `ops_config/` |
| `PHONO_OPS_MANIFESTS` | Explicit manifests to merge; otherwise selected from the manifest directory |
| `PHONO_DTYPE_SELECTIVE_BUILD` | Further prune dtype variants using the manifests; default: **OFF** |

PhonoP2C's export task produces per-model and merged operator manifests. Place the required manifests in `ops_config/` and configure and build again to register the union of their operators. Without manifests, the full kernel library is built.

Dtype pruning is independent of operator pruning. When enabled, the build generates `selected_op_variants.h` from the merged manifests and retains the corresponding dtype variants. Standard SDK builds and `pixi run config` explicitly disable dtype pruning. For a custom build, override it after running that task:

```sh
pixi run python tools/configure.py -DPHONO_DTYPE_SELECTIVE_BUILD=ON
pixi run build
```

## Model tests and performance measurement

Set `PHONO_TEST_MODEL_DIR` to a complete v2.2 model package before running CTest; otherwise model tests are skipped. Release validation requires them to execute using `PHONO_REQUIRE_TEST_MODEL=1`; see the SDK guide.

From the repository root, pass a model directory, iteration count, and warmup count (use `.exe` on Windows):

```sh
build/<preset>/bin/performance_benchmark <model-dir> 20 5
```

The program reports CSV measurements for model-load latency and RSS, individual model methods, generation across history and pinyin window lengths, incremental fill, and microbenchmarks for cache clearing, beam reordering, and session reset. Input construction, context filling, and cache initialization before each sample are excluded from its timing.
