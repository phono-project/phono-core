# phono-core

## 简介

phono-core 是 PhonoP2C 的边缘侧 C++ 推理引擎：在 ExecuTorch 运行时上部署两段式 PostfixLM 拼音转汉字模型，输入「中文上下文 + 拼音音节」，输出汉字候选。它只依赖 ExecuTorch 的 C++ API 与少量系统库，适合部署到移动端或嵌入式设备。

代码按功能划分为 src/core、src/algo、src/context、src/engine 四个子目录，对应 phono::core、phono::algo、phono::context、phono::engine 四个子命名空间，另有 src/custom_ops 目录（命名空间 phono::ops）提供 ExecuTorch 自定义算子。src/core 与 src/algo 不依赖 ExecuTorch，是纯逻辑层，可独立测试；src/context 与 src/engine 依赖 ExecuTorch 运行时。interface 目录实现面向 C-ABI 的调用规范，编译为共享库 libphono_core.so，供 C 语言、Python 绑定、移动端或其他平台直接调用。

## Introduction

phono-core is the on-device C++ inference engine for PhonoP2C: it deploys the two-stage PostfixLM pinyin-to-Chinese model on the ExecuTorch runtime, taking "Chinese context + pinyin syllables" as input and producing Chinese-character candidates. It depends only on ExecuTorch's C++ API and a few system libraries, making it suitable for mobile or embedded deployment.

The code is organized into four subdirectories — src/core, src/algo, src/context, src/engine — mirrored by the sub-namespaces phono::core, phono::algo, phono::context and phono::engine, plus src/custom_ops (namespace phono::ops) for ExecuTorch custom operators. src/core and src/algo are ExecuTorch-independent pure logic that can be tested in isolation; src/context and src/engine depend on the ExecuTorch runtime. The interface directory implements the C-ABI call convention and is compiled into the shared library libphono_core.so for direct consumption by C code, Python bindings, mobile or other platforms.

## 项目结构

- src/core — 与 ExecuTorch 无关的纯逻辑，命名空间 phono::core
  - config — 加载 v2.2 模型包；解析相互独立的 engine_config 与 context_manager 版本化 JSON
  - tokenizer — 汉字、上下文、拼音词表的编解码；精确查询与编辑距离查询显式分离
  - pinyin_normalizer — 大小写、分隔符和 `j/q/x/y + v` 规范化，并保留原输入坐标
  - utf8_util — UTF-8 逐字符切分工具（基于 uni-algo）
- src/algo — 与 ExecuTorch 无关的解码算法，命名空间 phono::algo
  - zh2hans — 繁体转简体（zh2Hans）最长匹配替换，字典由 res/zh2hans.json 配置期 codegen
  - pinyin_dag — 从拼音词表 Trie 枚举合法边，执行 Scorer-Viterbi 或 checked FMM
- src/context — 流式上下文状态，命名空间 phono::context
  - kv_cache — PersistentTensor：零拷贝的持久缓存缓冲区及其子视图
  - context — Context / ContextManager：B 路 self-KV Cache 视图、token id 序列，以及可复用上下文槽位管理
- src/engine — 推理引擎，命名空间 phono::engine
  - inference_engine — InferenceEngine 组合分词 Scorer、tokenizer 与 P2C 模型，提供稳定的自动分词入口；InferenceSession 提供无状态 fill / generate API
- src/custom_ops — ExecuTorch 自定义算子，命名空间 phono::ops：update_mhsa_kv
- interface — C-ABI 调用规范（phono_api.h / phono_api.cpp），编译为 libphono_core.so
- apps — 可执行程序：cli_demo_capi（C-ABI 智能分词演示）与 ime_demo_capi（交互式输入法演示）
- tests — 单元测试与模型级测试，由 CTest 驱动
- docs — 文档（docs/zh-cn 与 docs/en-us 分语言维护）
- third_party — ExecuTorch 源码，由 pixi 的 setup 任务拉取

## Project Layout

- src/core — ExecuTorch-independent pure logic, namespace phono::core
  - config — loads v2.2 packages and parses the independent, versioned engine and context-manager JSON configs
  - tokenizer — Chinese/context/pinyin encoding plus explicit exact and nearest pinyin lookup
  - pinyin_normalizer — case, separator, and `j/q/x/y + v` normalization with original-input coordinates
  - utf8_util — UTF-8 per-character splitting utilities (backed by uni-algo)
- src/algo — ExecuTorch-independent decoding algorithms, namespace phono::algo
  - zh2hans — Traditional->Simplified (zh2Hans) longest-match replacement; the table is code-generated from res/zh2hans.json at configure time
  - pinyin_dag — legal-edge enumeration from the vocabulary Trie, scorer-Viterbi, and checked FMM
- src/context — streaming context state, namespace phono::context
  - kv_cache — PersistentTensor: zero-copy persistent cache buffers and sub-views
  - context — Context / ContextManager: B-wide self-KV-cache views, token-id sequences, and reusable context slots
- src/engine — inference engine, namespace phono::engine
  - inference_engine — InferenceEngine composes the segmenter scorer, tokenizer, and P2C models behind a stable auto-segmentation API; InferenceSession provides stateless fill/generate
- src/custom_ops — ExecuTorch custom operator, namespace phono::ops: update_mhsa_kv
- interface — the C-ABI call convention (phono_api.h / phono_api.cpp), compiled into libphono_core.so
- apps — executables: cli_demo_capi (C-ABI smart-segmentation demo) and ime_demo_capi (interactive IME demo)
- tests — unit and model-level tests driven by CTest
- docs — documentation, maintained separately under docs/zh-cn and docs/en-us
- third_party — ExecuTorch source, fetched by the pixi setup task

## 模型包

模型包是一个自包含的目录，InferenceEngine 以该目录路径构造：

- config.json — v2.2 模型结构配置；可选的 segmenter 节指向分词模型和字符词表
- bins/pre_model.pte — 前段解码器，包含 pre_model_pass1、pre_model_cross_kv 与 pre_model_pass2
- bins/post_model.pte — 后段拼音编码器，输出 hidden states 与定宽候选 ID 表
- bins/pinyin_segment.pte — BHWC 间隙 Scorer（模型包启用智能分词时必需）
- vocabs/chinese_vocab.txt — 汉字词表（预测输出空间）
- vocabs/context_vocab.txt — 上下文词表（含特殊符号，如 bos_token）
- vocabs/pinyin_vocab.txt — 拼音音节词表（模型输入）
- vocabs/pinyin_char_vocab.txt — Scorer 字符词表（启用智能分词时必需）

本地准备的 v2.2 base 模型包为 `phonop2c_v2_2_base_w4a8_model` 与
`phonop2c_v2_2_base_w8a8_model`。w4a8/w8a8 描述 P2C 主模型；当前分词
Scorer 使用可动态长度执行的非量化 PTE。模型格式锁定为字符串 `"2.2"`，
不兼容旧包。运行期配置不放入模型包，使用仓库的
`core_configs/engine_config.json` 与 `core_configs/context_manager.json`。

Trie 在加载时直接由 `pinyin_vocab.txt` 构造，不在模型包中保存另一份词典结构。

## Model Package

A model package is a self-contained directory; InferenceEngine is constructed with its path:

- config.json — the v2.2 model structure; its optional segmenter section points to the scorer and character vocabulary
- bins/pre_model.pte — the decoder with pre_model_pass1, pre_model_cross_kv and pre_model_pass2
- bins/post_model.pte — the pinyin encoder, returning hidden states and a bounded candidate-ID table
- bins/pinyin_segment.pte — the BHWC gap scorer (required when smart segmentation is enabled)
- vocabs/chinese_vocab.txt — the Chinese-character vocabulary (prediction output space)
- vocabs/context_vocab.txt — the context vocabulary (including special tokens such as bos_token)
- vocabs/pinyin_vocab.txt — the pinyin-syllable vocabulary (model input)
- vocabs/pinyin_char_vocab.txt — scorer character vocabulary (required when smart segmentation is enabled)

The prepared local v2.2 packages are `phonop2c_v2_2_base_w4a8_model` and
`phonop2c_v2_2_base_w8a8_model`. w4a8/w8a8 describes the main P2C model; the
current segmentation scorer is an unquantized PTE that supports dynamic input
lengths. The package format is locked to string `"2.2"` and intentionally does
not accept old packages. Runtime policy lives outside the package in
`core_configs/engine_config.json` and `core_configs/context_manager.json`.

The Trie is built directly from `pinyin_vocab.txt` at load time, so no second
serialized dictionary structure is stored in the package.

## 构建与运行

前置要求：pixi 环境与 vcpkg。uni-algo 与 nlohmann-json 由 vcpkg.json 管理；繁体-简体（zh2Hans）规则在 res/zh2hans.json 中维护，由 pixi 环境里的 Python 在 CMake 配置期 codegen 成原生 C++ 表（codegen/zh2hans_codegen.py）。ExecuTorch 由 pixi 任务拉取源码后随主工程一起编译：

- pixi run setup — 将 ExecuTorch 克隆到 third_party/executorch，并只初始化本工程实际用到的子模块（XNNPACK 后端依赖 + flatbuffers/flatcc/json/gflags），避免拉取全部子模块；已存在时保留并仅修正子模块
- pixi run config — 配置 CMake 构建
- pixi run build — 增量编译

配置阶段可在 CMake 缓存中覆写选项：PHONO_USE_INSTALLED_EXECUTORCH 指定使用独立安装的 ExecuTorch（需同时设置 CMAKE_PREFIX_PATH）；EXECUTORCH_SOURCE_DIR 指定 ExecuTorch 源码路径（默认 third_party/executorch）；PHONO_XNNPACK_WEIGHT_CACHE 控制 XNNPACK delegate 是否共享 packed weights（默认 ON）。可执行文件与共享库默认输出到 results/ 目录。

### 选择性编译（算子裁剪）

PhonoP2C 的 export.py 在导出 pre_model.pte / post_model.pte 之后会生成 ExecuTorch 选择编译清单（selected_operators.yaml 格式）：每个模型一份，以及一份合并清单，记录模型实际使用的算子与精度（dtype/dim-order）。把这些清单文件复制到 ops_config/ 目录后重新 `pixi run config && pixi run build`，编译会自动裁剪 ExecuTorch 内核库：

- 算子裁剪：通过 EXECUTORCH_SELECT_OPS_LIST 只注册清单中出现的算子，避免链接完整 portable_ops_lib；
- 精度裁剪：由合并清单生成 selected_op_variants.h 并配合 EXECUTORCH_SELECTIVE_BUILD_DTYPE 只保留清单中出现的 dtype 变体（ExecuTorch 官方仅支持单个 .pte 模型走 dtype 裁剪，这里改为基于多模型合并清单）。

可调 CMake 选项：PHONO_OPS_CONFIG_DIR（清单目录，默认 ops_config/）、PHONO_OPS_MANIFESTS（手动指定要合并的清单文件，默认取 <tag>_ops.yaml 合并清单）、PHONO_DTYPE_SELECTIVE_BUILD（默认 ON，关闭则只做算子裁剪）。没有清单时自动回退为完整内核库构建。

不同平台的 vcpkg 配置、CMake 生成器与编译工具链有所差异，请参阅 docs/zh-cn/build.md（简体中文）与 docs/en-us/build.md（English）中的分平台说明。智能分词的 MAP 原理、非法输入策略和版本化 JSON Schema 见 docs/zh-cn/pinyin-segmentation.md；运行配置与 C ABI 分别见 core-config.md 和 c-api.md。

## Build & Run

Prerequisites: the pixi environment and vcpkg. uni-algo and nlohmann-json are declared in vcpkg.json; the Traditional->Simplified (zh2Hans) rules live in res/zh2hans.json and are compiled into a native C++ table by pixi's Python at CMake configure time (codegen/zh2hans_codegen.py). The pixi task fetches ExecuTorch's source and it is compiled together with this project:

- pixi run setup — clones ExecuTorch into third_party/executorch and initializes only the submodules this project actually needs (the XNNPACK backend deps plus flatbuffers/flatcc/json/gflags) instead of pulling all of them; skipped/fixed-up when already present
- pixi run config — configures the CMake build
- pixi run build — incremental build

At configure time, CMake cache variables can be overridden: PHONO_USE_INSTALLED_EXECUTORCH selects a separately installed ExecuTorch (set CMAKE_PREFIX_PATH accordingly), EXECUTORCH_SOURCE_DIR points to the ExecuTorch source (default third_party/executorch), and PHONO_XNNPACK_WEIGHT_CACHE controls packed-weight sharing across XNNPACK delegates (default ON). Executables and the shared library are output to results/ by default.

### Selective build (operator pruning)

PhonoP2C's export.py emits ExecuTorch selective-build manifests (selected_operators.yaml format) after exporting pre_model.pte / post_model.pte: one per model plus a merged one, recording exactly which operators and dtypes (dtype/dim-order kernel variants) the models use. Copy the manifests into ops_config/ and re-run `pixi run config && pixi run build` to prune the ExecuTorch kernel library:

- Operator pruning: EXECUTORCH_SELECT_OPS_LIST registers only the operators present in the manifests, so the full portable_ops_lib is never linked.
- Dtype (precision) pruning: a selected_op_variants.h header is generated from the merged manifest and combined with EXECUTORCH_SELECTIVE_BUILD_DTYPE to keep only the dtype variants actually used. (Upstream ExecuTorch only supports dtype-selective-build from a single .pte model; here it is driven by the multi-model merged manifest instead.)

Tunable CMake options: PHONO_OPS_CONFIG_DIR (manifest directory, default ops_config/), PHONO_OPS_MANIFESTS (explicit manifest list to merge; default picks the <tag>_ops.yaml merged manifest), and PHONO_DTYPE_SELECTIVE_BUILD (default ON; set OFF to do operator pruning only). When no manifest is present the build falls back to the full kernel library.

Because the vcpkg configuration, the CMake generator and the compiler toolchain differ per platform, see docs/en-us/build.md (English) or docs/zh-cn/build.md (简体中文) for platform-specific instructions. The MAP derivation, invalid-input behavior, and versioned JSON schema are in docs/en-us/pinyin-segmentation.md; runtime configuration and the C ABI are documented in core-config.md and c-api.md.

## 使用

下载模型之后，运行 C-ABI 演示程序：

```
results/cli_demo_capi phonop2c_v2_2_base_w4a8_model
```

如需排除交互输入并进行可重复的性能测量，可向 CSV 基准程序传入模型包、采样次数和预热次数：

```
results/performance_benchmark phonop2c_v2_2_base_w4a8_model 20 5
```

该程序报告模型加载耗时与 RSS、pre/post 各方法耗时、不同历史及拼音窗口长度下的生成耗时、单 token 增量 fill 耗时，以及 KV 清零、全量/增量 beam 重排和 session reset 的微基准。每个计时样本之前的输入构造、上下文填充和缓存初始化不计入样本耗时。

CLI 从 stdin 读取连续拼音窗口，例如 `nihaoma`。它通过版本化 JSON 调用稳定的自动分词接口，展示 `scorer_viterbi`/`checked_fmm` 路由、非法位置和规范化事件，再把响应中的 `pinyin_ids` 直接送入生成接口。第二、第三个参数可分别指定 `engine_config.json` 和 `context_manager.json`；缺省时读取仓库 `core_configs/` 下的同名文件。

模拟实时输入法编辑，可以运行：
```
results/ime_demo_capi phonop2c_v2_2_base_w4a8_model
```
程序支持左右移动光标以及 Backspace/Delete 删除；拼音为空时，这些编辑键作用于已确认历史。切分栏实时显示智能/FMM 路由与非法范围，下一栏显示当前历史，其余栏显示候选。前两个可选参数仍是两份 JSON，第四个参数可指定上下文数量（默认 2）。按 Ctrl-C 退出。

## Usage

Run the C-ABI demo after downloading the model:

```
results/cli_demo_capi phonop2c_v2_2_base_w4a8_model
```

For repeatable performance measurements without interactive I/O, run the CSV
benchmark with a model package, iteration count and warmup count:

```
results/performance_benchmark phonop2c_v2_2_base_w4a8_model 20 5
```

It reports model-load RSS and latency, individual pre/post method latency,
generation across several history/window lengths, and one-token incremental
fill latency. Setup and cache initialization are outside each timed sample.

The CLI reads one continuous pinyin window per line, such as `nihaoma`. It calls the stable automatic segmentation endpoint with versioned JSON, displays the `scorer_viterbi`/`checked_fmm` route, invalid ranges, and normalization events, then passes the returned `pinyin_ids` directly to generation. Optional arguments two and three select `engine_config.json` and `context_manager.json`; the defaults come from the repository's `core_configs/` directory.

For real-time editing with left/right movement and backspace/delete, run:

```
results/ime_demo_capi phonop2c_v2_2_base_w4a8_model
```

When pinyin is empty, editing keys operate on committed history. The segmentation bar displays the smart/FMM route and invalid ranges, followed by history and candidate rows. The first two optional arguments are the engine and context JSON paths; a fourth argument selects the context count (default 2). Press Ctrl-C to exit.

## 流式推理设计

InferenceSession 是无状态的：会话状态不保存在会话内部，而是显式地保存在调用方传入的上下文槽位中，因此一个会话可以驱动多个上下文，也便于在多个槽位之间切换。每个上下文槽位持有三样东西：KV Cache 的 PersistentTensor 视图、已提交文本的 token id 序列、以及游标 current_seqlen（下一次写入的缓存位置）。

- fill 在历史未改变时复用缓存；历史追加时只运行新增区间，并将 beam 0 的新切片复制到其他 beam。历史长度受 max_history_length 软上限约束，一旦超出便截断到 max_history_length - slack_interval（保留第一个 BOS 作为注意力汇点）。
- generate 只读取严格因果历史和当前生成的临时 cross-attention 缓存；生成期间不改变 history_seqlen，只推进 current_seqlen。generate 之前若历史达到 max_history_length，先按相同规则截断；拼音窗口长度受 max_pinyin_length 约束。
- 中止通过「回调函数 + 上下文指针」检查：generate 在每一步之间轮询回调，回调返回真时中止并回滚游标（current_seqlen 恢复到 history_seqlen），下一次 generate 可以从已提交的历史继续。
- ContextManager 用 core_config 配置 beam、slack_interval、min_accept_context、max_context_length、max_history_length、max_pinyin_length 与槽位淘汰参数，支持前缀复用、BOS 保护的左移窗口复用和直接重算。

详细的架构说明见 docs/zh-cn/architecture.md。

## Streaming Inference Design

InferenceSession is stateless: per-conversation state lives in the context slot passed explicitly by the caller, so one session can drive many contexts and switch freely between slots. Each context slot holds three things: PersistentTensor views into the KV caches, the token-id sequence of the committed text, and the cursor current_seqlen (the cache write position for the next step).

- fill reuses unchanged history; when history is appended it runs only the new range and copies beam zero's new slice to the other beams. History is capped by the soft limit max_history_length; once exceeded it is truncated to max_history_length minus slack_interval, keeping the first BOS as the attention sink.
- generate reads strictly-causal history plus the temporary cross-attention generation cache and never changes history_seqlen during generation, only advancing current_seqlen. Before generating, if the history has reached max_history_length it is truncated with the same rule, and the pinyin window is capped at max_pinyin_length.
- Cancellation is a callback-plus-context-pointer check: generate polls the callback between steps, and when it returns true generation aborts and the cursors are rolled back (current_seqlen is restored to history_seqlen), so the next generate can resume from the committed history.
- ContextManager takes beam, slack_interval, min_accept_context, max_context_length, max_history_length, max_pinyin_length and slot-eviction parameters from the core_config, supporting prefix reuse, BOS-protected left-shift reuse and direct recalculation.

See docs/en-us/architecture.md for the full design.

## 测试

构建后执行 ctest：

```
ctest --test-dir build --output-on-failure
```

测试集包含三组：phono_core_tests（tokenizer、KV 缓存切片、上下文槽位复用、core_config 解析与校验）、phono_capi_tests（C-ABI 错误码表面，无需模型）、phono_engine_tests（模型级会话测试：取消回滚、拼音上限、历史窗口截断）。phono_engine_tests 需要设置 PHONO_TEST_MODEL_DIR 指向 v2 模型包目录才会真正执行，未设置时自动跳过。

## Testing

After building, run CTest:

```
ctest --test-dir build --output-on-failure
```

The suite has three test groups: phono_core_tests (tokenizer, KV-cache slicing, context slot reuse, core_config parsing and validation), phono_capi_tests (C-ABI error-code surfaces, no model required) and phono_engine_tests (model-level session tests: cancellation rollback, pinyin limit, history windowing). phono_engine_tests only runs when PHONO_TEST_MODEL_DIR points to a v2 model package directory; otherwise it is skipped.

## 许可证与最终声明

由于本人精力有限，对架构、标准的设计无法全力，而且文档大部分使用了 LLM 生成，不可避免的会存在纰漏、更新不及时等问题。如果发现有任何问题，欢迎提出 issues。

本项目基于 Apache License 2.0 开源，详见 [LICENSE](LICENSE)。

## License and Final Note

Due to my limited time and resources, I am unable to devote my full attention to architectural and standard design. Furthermore, since most of the documentation was generated by an LLM, there will inevitably be some inaccuracies and delays in updates. If you find any issues, please feel free to open an issue.

Open-sourced under the Apache License 2.0; see [LICENSE](LICENSE).
