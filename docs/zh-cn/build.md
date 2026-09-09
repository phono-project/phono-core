# 构建指南

## 依赖

phono-core 的构建需要以下组件：

- pixi：用于创建统一的构建环境（提供 Python、CMake、Ninja 与 C/C++ 编译器，并负责拉取 ExecuTorch 的 Python 依赖）。Python 还在 CMake 配置期运行 codegen，把繁体-简体（zh2Hans）规则编译成原生 C++ 表。
- vcpkg：提供 uni-algo 与 nlohmann-json 两个 C++ 依赖，由仓库根目录的 vcpkg.json 声明。
- 网络连接：pixi 需要下载 Python 包；pixi run setup 需要克隆 ExecuTorch 源码（含子模块）。

## 通用流程

在仓库根目录依次执行：

```
pixi run setup
pixi run config
pixi run build
```

- setup：将 ExecuTorch 克隆到 third_party/executorch，并只初始化本工程实际用到的子模块（XNNPACK 后端依赖以及 third-party/flatbuffers、flatcc、json、gflags），不再拉取全部子模块；若已存在则保留并仅修正子模块。
- config：配置 CMake 构建，等价于 `cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release $CMAKE_ARGS`。可通过环境变量 CMAKE_ARGS 追加自定义参数。
- build：增量编译，等价于 `cmake --build build`。

CMake 在未显式指定 CMAKE_TOOLCHAIN_FILE 时会自动读取环境变量 VCPKG_ROOT，并拼接出 vcpkg 的 toolchain 路径；因此使用 vcpkg 时只需要在配置前导出 VCPKG_ROOT 即可，无需手工传参。若想使用独立安装的 ExecuTorch，可在配置时传入 -DPHONO_USE_INSTALLED_EXECUTORCH=ON 并设置 CMAKE_PREFIX_PATH；ExecuTorch 源码路径由 EXECUTORCH_SOURCE_DIR 指定（默认 third_party/executorch）。

构建产物默认输出到 results/ 目录，包括：

- libphono_core.so — C-ABI 共享库
- cli_demo_capi — C-ABI 智能分词演示程序
- ime_demo_capi — 交互式 C-ABI 输入法演示程序

## 选择性编译（算子/精度裁剪）

PhonoP2C 的 export.py 在导出后会把 ExecuTorch 选择编译清单（每模型一份 + 合并一份）写入
PhonoP2C/export_output/manifests/。要把 ExecuTorch 内核库裁剪到模型实际用到的算子与精度，
把清单文件复制到 phono-core 的 ops_config/ 目录后重新配置即可：

```
cp <PhonoP2C>/export_output/manifests/<tag>_ops.yaml ops_config/
pixi run config
pixi run build
```

- 算子裁剪：EXECUTORCH_SELECT_OPS_LIST 只注册清单中出现的算子，完整 portable_ops_lib 不再参与链接。
- 精度裁剪：由合并清单生成 selected_op_variants.h，并配合 EXECUTORCH_SELECTIVE_BUILD_DTYPE 只保留实际用到的 dtype 变体。ExecuTorch 官方只支持基于单个 .pte 模型的 dtype 选择编译，这里改为基于多模型合并清单驱动。

相关 CMake 选项：

- PHONO_OPS_CONFIG_DIR（默认 `ops_config/`）：清单读取目录。
- PHONO_OPS_MANIFESTS：显式指定需要合并的清单文件（分号分隔）；默认取 PHONO_OPS_CONFIG_DIR 下的 `<tag>_ops.yaml` 合并清单，若不存在则合并全部 `*_ops.yaml`。
- PHONO_DTYPE_SELECTIVE_BUILD（默认 ON）：置为 OFF 则只做算子裁剪。
- PHONO_USE_INSTALLED_EXECUTORCH、EXECUTORCH_SOURCE_DIR：保持不变。

当 ops_config 中没有清单时，构建自动回退为完整内核库（原有行为）。

## Linux

前置：安装 pixi；安装 vcpkg 并导出 VCPKG_ROOT；系统装有与 pixi 兼容的编译工具链或由 pixi 提供。

```
# 安装 vcpkg（示例路径 ~/vcpkg）
git clone https://github.com/microsoft/vcpkg ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh

# 配置与编译
export VCPKG_ROOT=$HOME/vcpkg
pixi run setup
pixi run config
pixi run build
```

pixi 的 linux-64 平台（linux-glibc228）使用 glibc 2.28 作为目标，生成的二进制可部署到较老的发行版。请保持 VCPKG_ROOT 导出，vcpkg 会自动下载编译 uni-algo；构建还要求 pixi 环境提供 Python3，供配置期 codegen 使用。

## macOS

前置：安装 pixi 与 Xcode Command Line Tools（提供 clang）；安装 vcpkg。

pixi 的 osx-arm64 平台面向 Apple Silicon。步骤与 Linux 基本一致：

```
git clone https://github.com/microsoft/vcpkg $HOME/vcpkg
$HOME/vcpkg/bootstrap-vcpkg.sh

export VCPKG_ROOT=$HOME/vcpkg
pixi run setup
pixi run config
pixi run build
```

生成的动态库名与 Linux 不同（macOS 为 libphono_core.dylib），CMake 会自动处理。Xcode 版本过旧时，ExecuTorch 的编译可能失败，请升级到较新的 Xcode 与 Command Line Tools。

## Windows

前置：安装 pixi；安装 vcpkg（建议放在 C:\vcpkg）；安装 Visual Studio 2022 的 C++ 工作负载，或者由 pixi 提供 MSVC 工具链。请在开发者 PowerShell（Developer PowerShell）中执行，以便 CMake 能找到 MSVC 环境。

```
# 安装 vcpkg（PowerShell）
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat

# 配置与编译
$env:VCPKG_ROOT="C:\vcpkg"
pixi run setup
pixi run config
pixi run build
```

Windows 上 CMake 会定义 NOMINMAX 与 WIN32_LEAN_AND_MEAN，避免 windows.h 的宏污染；MSVC 运行库固定为多线程静态运行时（MT/MTd），以消除 DLL 依赖。动态库输出为 phono_core.dll。若使用 Ninja + MSVC，请确保在开发者 PowerShell 中运行 pixi run config，否则可能找不到 cl.exe。

## 常见问题

- ExecuTorch 克隆失败：检查网络与代理设置，删除 third_party/executorch 后重试 pixi run setup。
- zh2Hans codegen 失败：确认在 pixi 环境中运行 pixi run config（配置期会调用 Python3 执行 codegen/zh2hans_codegen.py）；也可直接运行 `python codegen/zh2hans_codegen.py --input res/zh2hans.json --output src/gen/zh2hansdict.h` 复现。
- 模型加载失败：模型包必须是 v2.2 格式（`model_format_version` 为字符串 `"2.2"`），pre 程序包含 `pre_model_pass1`、`pre_model_cross_kv` 与 `pre_model_pass2`；声明了 segmenter 时还必须提供对应 PTE 和字符词表。
