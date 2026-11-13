# SDK、兼容性与发布 / SDK, compatibility and releases

phono-core 独立维护版本，唯一来源是根目录 `VERSION.txt`。Release 标签必须为对应的 `vX.Y.Z`，前端通过普通提交更新 SDK 版本，不要求同步前端版本，也不触发跨仓库发布。

phono-core owns its version, with the root `VERSION.txt` as the single source. Release tags must match `vX.Y.Z`. Frontends update their SDK dependency in ordinary commits; their versions are independent and no cross-repository release is triggered.

## 使用 SDK / Consume the SDK

从 [GitHub Releases](https://github.com/phono-project/phono-core/releases) 下载对应版本的平台 SDK，解压到任意 SDK 前缀。正式发布覆盖以下五个平台，模型不随 SDK 打包。

Download the platform SDK for the desired version from [GitHub Releases](https://github.com/phono-project/phono-core/releases) and extract into an SDK prefix. Releases cover the following five platforms. Models are distributed separately.

| 平台 / Platform | Release 资产 / Release asset |
| --- | --- |
| Linux x86_64 | `phono-core-sdk-X.Y.Z-linux-x86_64.tar.gz` |
| Linux ARM64 | `phono-core-sdk-X.Y.Z-linux-arm64.tar.gz` |
| Windows x64 | `phono-core-sdk-X.Y.Z-windows-x64.zip` |
| macOS ARM64 | `phono-core-sdk-X.Y.Z-macos-arm64.tar.gz` |
| Android ARM64 | `phono-core-sdk-X.Y.Z-android-armv8.tar.gz` |

SDK 包含 `include/phono-core/phono_api.h`、Linux 的 `lib/libphono_core.so` 或 Windows 的 `bin/phono_core.dll` 和 `lib/phono_core.lib`、`lib/cmake/phono-core`、`share/licenses/phono-core`、`share/phono-core/VERSION` 及 `build-info.json`。安装后的 C API 可以由纯 C 工程链接，不要求消费者安装 ExecuTorch、Python 或 core 源码。

The SDK contains `include/phono-core/phono_api.h`, Linux `lib/libphono_core.so` or Windows `bin/phono_core.dll` and `lib/phono_core.lib`, `lib/cmake/phono-core`, `share/licenses/phono-core`, `share/phono-core/VERSION`, and `build-info.json`. A C-only project can link the installed C API without installing ExecuTorch, Python, or core sources.

```cmake
find_package(phono-core 0.1.3 CONFIG REQUIRED)
target_link_libraries(frontend PRIVATE phono::core)
```

配置消费者时通过 `-DCMAKE_PREFIX_PATH=<SDK-prefix>` 指定 SDK。Windows 安装时将 DLL 放在应用目录，Linux 前端按自己的发行版和安装布局部署共享库。

Configure consumers with `-DCMAKE_PREFIX_PATH=<SDK-prefix>`. On Windows, deploy the DLL alongside the application. Linux frontends deploy the shared library according to their distribution and installation layout.

## 兼容范围 / Compatibility

公开接口是 `phono_api.h` 中的 C ABI，构建信息记录其 ABI 代数为 1。对象通过不透明句柄传递；字符串、数组和候选结果须用对应的 `phono_free` 或结果释放函数释放。配置 JSON 的版本与模型格式版本分别校验，C++ 内部类不属于 SDK 接口。修改现有导出函数、枚举值或公开结构布局属于 ABI 变更，应更新版本并重新验证前端。

The public interface is the C ABI in `phono_api.h`; build information records ABI generation 1. Objects use opaque handles, and returned strings, arrays, and candidate results must be released with the matching `phono_free` or result-release function. Configuration JSON versions and model format versions are checked separately. Internal C++ classes are outside the SDK interface. Changing existing exports, enum values, or public structure layouts changes the ABI and requires a version update and frontend verification.

当前模型格式为 `2.2`，正式交付验证两个模型仓库：[W4A8](https://huggingface.co/afirelily/phonop2c_v2_2_base_w4a8_model) 的 `c11bc50c13a5a6ff627ab60a3f5571b8fdbacc73` 和 [W8A8](https://huggingface.co/afirelily/phonop2c_v2_2_base_w8a8_model) 的 `cf11a953a62fbba10f0eb585bc65802e01b70f7d`。配置、前后段 `.pte`、词表及配置启用的分词模型必须完整。下载脚本写入本仓库 `cache/models`，测试显式接收模型目录。

The current model format is `2.2`. Release verification covers two snapshots: [W4A8](https://huggingface.co/afirelily/phonop2c_v2_2_base_w4a8_model) at `c11bc50c13a5a6ff627ab60a3f5571b8fdbacc73` and [W8A8](https://huggingface.co/afirelily/phonop2c_v2_2_base_w8a8_model) at `cf11a953a62fbba10f0eb585bc65802e01b70f7d`. Configuration, pre/post `.pte` files, vocabularies, and any enabled segmenter assets must be present. The downloader writes into this repository's `cache/models`; tests receive model directories explicitly.

一个通用运行库同时支持这两种量化模型，不发布按精度拆分的运行库。ExecuTorch 提交和子模块由 `ci/core-dependencies.json` 锁定，Python 与构建环境由已提交的 `pixi.lock` 和 `ci/pixi.lock` 锁定。运行库包含 XNNPACK、`ops_config` 清单的算子并集及 `phono::update_mhsa_kv.out`，关闭 dtype 裁剪。新模型若需要清单外算子，需在 core 中更新清单并重新构建验证。

One universal runtime supports both quantized models; no precision-specific runtimes are published. `ci/core-dependencies.json` pins the ExecuTorch commit and gitlinks, while the committed `pixi.lock` and `ci/pixi.lock` pin Python and build environments. The runtime contains XNNPACK, the operator union from `ops_config`, and `phono::update_mhsa_kv.out`, with dtype pruning disabled. Models requiring additional operators need an updated core manifest and a rebuilt, verified runtime.

Windows x64 使用静态 MSVC CRT。macOS ARM64 构建环境要求 macOS 14 或更高版本；Android ARM64 使用 NDK 交叉编译，目标为 API 26、`arm64-v8a` 和 `c++_shared`。Linux x86_64 与 ARM64 分别使用对应的 Pixi 构建环境。

Windows x64 uses the static MSVC CRT. The macOS ARM64 build environment requires macOS 14 or later. Android ARM64 is cross-compiled with the NDK for API 26, `arm64-v8a`, and `c++_shared`. Linux x86_64 and ARM64 use their respective Pixi build environments.

## 模型包布局 / Model package layout

引擎加载模型包目录中的 `config.json`、`bins/` 与 `vocabs/`。当前格式版本为字符串 `"2.2"`，不兼容旧格式。运行策略另由 `engine_config.json` 与 `context_manager.json` 提供，见 [中文](zh-cn/core-config.md) / [English](en-us/core-config.md)。

The engine loads `config.json`, `bins/`, and `vocabs/` from the package directory. The current format version is the string `"2.2"`; older formats are not compatible. Runtime policy is supplied separately through `engine_config.json` and `context_manager.json`; see the [configuration guide](en-us/core-config.md).

| 路径 / Path | 内容 / Contents |
| --- | --- |
| `config.json` | 模型结构及可选分词器配置 / Model structure and optional segmenter configuration |
| `bins/pre_model.pte` | 解码器，含 `pre_model_pass1`、`pre_model_cross_kv`、`pre_model_pass2` / Decoder with those three methods |
| `bins/post_model.pte` | 拼音编码器，输出隐状态与定宽候选 ID 表 / Pinyin encoder returning hidden states and a bounded candidate-ID table |
| `vocabs/chinese_vocab.txt` | 汉字输出词表 / Chinese output vocabulary |
| `vocabs/context_vocab.txt` | 上下文基础词表 / Base context vocabulary |
| `vocabs/pinyin_vocab.txt` | 拼音音节词表 / Pinyin syllable vocabulary |
| `bins/pinyin_segment.pte` | 启用智能分词时使用的 BHWC 间隙 Scorer / BHWC gap scorer for smart segmentation |
| `vocabs/pinyin_char_vocab.txt` | 分词 Scorer 的字符词表 / Character vocabulary for the segmentation scorer |

`config.json` 中声明的 `bos_token` 等特殊符号在运行时追加到上下文词表。W4A8 / W8A8 表示 P2C 主模型的量化方式；当前分词 Scorer 使用支持动态输入长度的非量化 PTE。分词模型配置要求见 [中文](zh-cn/pinyin-segmentation.md) / [English](en-us/pinyin-segmentation.md)。

Special symbols such as `bos_token`, declared in `config.json`, are appended to the context vocabulary at runtime. W4A8 / W8A8 describes the main P2C model's quantization; the current segmentation scorer uses an unquantized PTE with dynamic input lengths. See the [segmentation guide](en-us/pinyin-segmentation.md) for configuration requirements.

## 构建与验证 / Build and verify

Windows 在 x64 Visual Studio 开发者终端中设置 `VCPKG_ROOT`，然后使用已锁定环境。Linux 使用独立的已锁定 sysroot 构建环境；两个入口都是 CI 和本地复现共用的仓库脚本。

On Windows, set `VCPKG_ROOT` in an x64 Visual Studio developer shell and use the locked environment. Linux uses its separate locked sysroot toolchain. Both entry points are repository scripts shared by CI and local reproduction.

```powershell
pixi run --locked python tools/build_sdk.py --prefix build/sdk --build-dir build/windows-sdk
```

```sh
pixi run --locked --manifest-path ci/pixi.toml python tools/ci_build_core.py --source . --prefix build/sdk --build-dir build/linux-sdk
```

PR、手动运行和 Release 工作流均在 Linux x86_64、Linux ARM64、Windows x64、macOS ARM64 上构建，并分别使用 W4A8、W8A8 模型目录运行 CTest。Linux x86_64 任务另外交叉编译并安装 Android ARM64 SDK；Android 产物不在此工作流中运行目标设备测试。

PR, manual, and release workflows build Linux x86_64, Linux ARM64, Windows x64, and macOS ARM64, running CTest separately with the W4A8 and W8A8 model directories. The Linux x86_64 job also cross-compiles and installs the Android ARM64 SDK; this workflow does not run Android device tests.

## 维护流程 / Maintainer workflow

完成代码、依赖及版本更新并合并到默认分支后，创建 `vX.Y.Z` 标签，再在 GitHub 发布对应 Release（正式版或预发布均可）。工作流完成构建、测试和 SDK 打包后，发布任务检出该标签及完整 Git 历史，校验版本与默认分支归属，将 CI 产物下载到 `dist/`，再显式上传上述五个平台的 SDK 压缩包。只推送标签不会触发交付；只有上传任务拥有写权限。

After merging code, dependency, and version updates into the default branch, create a `vX.Y.Z` tag and publish its GitHub Release, either stable or prerelease. After building, testing, and packaging, the release job checks out the tag with full Git history, validates the version and default-branch ancestry, downloads CI artifacts into `dist/`, and explicitly uploads the SDK archives for all five platforms above. Pushing a tag alone does not deliver artifacts; only the upload job has write permission.

演示程序保留为源码示例，按[构建指南](zh-cn/build.md)本地编译，不随 Release 发布。

Demos remain source examples for [local builds](en-us/build.md) and are not published in Releases.
