# phono-core

phono-core 是 PhonoP2C 的 C++ 推理引擎，根据中文上文和拼音生成汉字候选，支持连续拼音自动分词、上下文复用和取消生成。项目基于 ExecuTorch，提供 C ABI，供输入法和其他应用集成。目前提供 Linux x86_64 和 Windows x64 的 SDK。

phono-core is the C++ inference engine for PhonoP2C. It generates Chinese candidates from preceding Chinese text and pinyin, with automatic segmentation of continuous pinyin, context reuse, and generation cancellation. Built on ExecuTorch, it exposes a C ABI for input methods and other applications. SDKs are available for Linux x86_64 and Windows x64.

[PhonoP2C](https://github.com/phono-project/PhonoP2C) 提供模型训练与导出工具；[Euphony](https://github.com/phono-project/Euphony) 是使用本引擎的 Windows 输入法。

[PhonoP2C](https://github.com/phono-project/PhonoP2C) provides model training and export tools; [Euphony](https://github.com/phono-project/Euphony) is a Windows input method built with this engine.

## 快速体验 / Try the engine

按[构建指南](docs/zh-cn/build.md)从源码编译演示程序。另下载一个完整的 v2.2 模型包：[W4A8](https://huggingface.co/afirelily/phonop2c_v2_2_base_w4a8_model) 或 [W8A8](https://huggingface.co/afirelily/phonop2c_v2_2_base_w8a8_model)。安装 Hugging Face CLI 后，例如：

Build the demos from source using the [build guide](docs/en-us/build.md). Download a complete v2.2 model package separately: [W4A8](https://huggingface.co/afirelily/phonop2c_v2_2_base_w4a8_model) or [W8A8](https://huggingface.co/afirelily/phonop2c_v2_2_base_w8a8_model). With the Hugging Face CLI installed, for example:

```sh
hf download afirelily/phonop2c_v2_2_base_w4a8_model --local-dir ./phonop2c_v2_2_base_w4a8_model
```

在仓库根目录运行，将 `<model-dir>` 替换为模型目录的绝对路径。

Run from the repository root, replacing `<model-dir>` with the absolute path to the model.

Linux:

```sh
./build/bin/cli_demo_capi <model-dir>
```

Windows PowerShell:

```powershell
.\build\bin\cli_demo_capi.exe "<model-dir>"
```

启动后输入连续拼音（例如 `nihaoma`）并回车，程序显示拼音切分结果和汉字候选。将命令中的 `cli_demo_capi` 替换为 `ime_demo_capi` 可体验连续输入和编辑：左右键移动光标，Backspace / Delete 删除，Ctrl-C 退出。

Enter continuous pinyin such as `nihaoma` and press Enter to see segmentation and Chinese candidates. Replace `cli_demo_capi` with `ime_demo_capi` to try interactive input and editing: move the cursor with the arrow keys, delete with Backspace / Delete, and exit with Ctrl-C.

## 接入项目 / Integrate into an application

从 Releases 下载对应平台的 `phono-core-sdk-X.Y.Z-<platform>`，按 [SDK 说明](docs/sdk-release.md)解压。配置应用时通过 `CMAKE_PREFIX_PATH` 指定 SDK 目录，然后链接：

Download `phono-core-sdk-X.Y.Z-<platform>` from Releases and extract it as described in the [SDK guide](docs/sdk-release.md). Set `CMAKE_PREFIX_PATH` to the SDK directory when configuring your application, then link:

```cmake
find_package(phono-core CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE phono::core)
```

公开接口位于 `phono_api.h`。调用流程与资源释放要求见 [C API（中文）](docs/zh-cn/c-api.md) / [C API (English)](docs/en-us/c-api.md)。Windows 应用需随程序部署 SDK 的 `phono_core.dll`；Linux 应用按安装布局部署共享库。平台要求、ABI 兼容范围及 SDK 布局见 [SDK 说明](docs/sdk-release.md)。

The public interface is in `phono_api.h`. See the [C API guide](docs/en-us/c-api.md) for the call sequence and resource ownership. Windows applications must deploy the SDK's `phono_core.dll` alongside the executable; Linux applications deploy the shared library according to their installation layout. See the [SDK guide](docs/sdk-release.md) for platform requirements, ABI compatibility, and SDK layout.

## 模型与运行配置 / Models and runtime configuration

当前支持 v2.2 模型包，模型目录应包含 `config.json`、`bins/` 和 `vocabs/`。每个平台的同一运行库支持 W4A8 和 W8A8 两种模型。模型包格式和兼容要求见 [SDK 说明](docs/sdk-release.md)。

The engine supports v2.2 model packages containing `config.json`, `bins/`, and `vocabs/`. Each platform's runtime supports both W4A8 and W8A8 models. See the [SDK guide](docs/sdk-release.md) for the package layout and compatibility requirements.

运行参数通过 `engine_config.json` 和 `context_manager.json` 配置，分别控制分词与生成、上下文管理。演示包在 `share/phono-core/core_configs/` 中提供默认配置，源码仓库中位于 `core_configs/`。参数说明见 [中文](docs/zh-cn/core-config.md) / [English](docs/en-us/core-config.md)。

Runtime settings are split between `engine_config.json` for segmentation and generation, and `context_manager.json` for context management. Demo packages include defaults under `share/phono-core/core_configs/`; the source checkout keeps them in `core_configs/`. See the [configuration guide](docs/en-us/core-config.md) for parameters.

## 源码构建与测试 / Build and test from source

源码构建需要 Pixi 和平台对应的 C++ 工具链。环境准备与构建选项见 [中文构建指南](docs/zh-cn/build.md) / [English build guide](docs/en-us/build.md)。在准备好的开发终端中，从仓库根目录运行：

Source builds require Pixi and a C++ toolchain for your platform. See the [build guide](docs/en-us/build.md) for environment setup and build options. From a prepared developer shell in the repository root, run:

```sh
pixi install --locked
pixi run setup
pixi run config
pixi run build
pixi run test
```

模型级测试需将 `PHONO_TEST_MODEL_DIR` 指向完整的 v2.2 模型包；未设置时这些测试会跳过。编译后的演示程序位于 `build/bin/`，从仓库根目录运行 `build/bin/cli_demo_capi <model-dir>`（Windows 使用 `.exe`），默认读取 `core_configs/`。性能测试命令见构建指南。

Set `PHONO_TEST_MODEL_DIR` to a complete v2.2 model package to run model tests; otherwise those tests are skipped. Built demos are in `build/bin/`. Run `build/bin/cli_demo_capi <model-dir>` from the repository root (use `.exe` on Windows) to load defaults from `core_configs/`. Benchmark instructions are in the build guide.

## 代码与详细文档 / Code and documentation

| 路径 / Path | 用途 / Purpose |
| --- | --- |
| `src/core/`, `src/algo/` | 配置、词表与分词算法 / Configuration, vocabularies, and segmentation |
| `src/context/`, `src/engine/`, `src/custom_ops/` | 上下文管理、推理和自定义算子 / Context management, inference, and custom operators |
| `interface/` | 公开 C ABI / Public C ABI |
| `apps/`, `tests/` | 演示、基准与测试 / Demos, benchmarks, and tests |

- 架构与流式推理 / Architecture and streaming inference: [中文](docs/zh-cn/architecture.md) · [English](docs/en-us/architecture.md)
- 拼音分词 / Pinyin segmentation: [中文](docs/zh-cn/pinyin-segmentation.md) · [English](docs/en-us/pinyin-segmentation.md)
- SDK 兼容性与发布 / SDK compatibility and releases: [指南 / Guide](docs/sdk-release.md)

## 许可证与反馈 / License and feedback

本项目使用 [Apache License 2.0](LICENSE)。问题与建议请提交至 [Issues](https://github.com/phono-project/phono-core/issues)。

Licensed under [Apache License 2.0](LICENSE). Report problems and suggestions through [Issues](https://github.com/phono-project/phono-core/issues).
