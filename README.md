# phono-core

## 简介

phono-core 是 PhonoP2C 的边缘侧 C++ 推理引擎：在 ExecuTorch 运行时上部署两段式 PostfixLM 拼音转汉字模型，输入「中文上下文 + 拼音音节」，输出汉字候选。它只依赖 ExecuTorch 的 C++ API 与少量系统库，适合部署到移动端或嵌入式设备。

代码按功能划分为 src/core、src/algo、src/context、src/engine 四个子目录，对应 phono::core、phono::algo、phono::context、phono::engine 四个子命名空间，另有 src/custom_ops 目录（命名空间 phono::ops）提供 ExecuTorch 自定义算子。src/core 与 src/algo 不依赖 ExecuTorch，是纯逻辑层，可独立测试；src/context 与 src/engine 依赖 ExecuTorch 运行时。interface 目录实现面向 C-ABI 的调用规范，编译为共享库 libphono_core.so，供 C 语言、Python 绑定、移动端或其他平台直接调用。

## Introduction

phono-core is the on-device C++ inference engine for PhonoP2C: it deploys the two-stage PostfixLM pinyin-to-Chinese model on the ExecuTorch runtime, taking "Chinese context + pinyin syllables" as input and producing Chinese-character candidates. It depends only on ExecuTorch's C++ API and a few system libraries, making it suitable for mobile or embedded deployment.

The code is organized into four subdirectories — src/core, src/algo, src/context, src/engine — mirrored by the sub-namespaces phono::core, phono::algo, phono::context and phono::engine, plus src/custom_ops (namespace phono::ops) for ExecuTorch custom operators. src/core and src/algo are ExecuTorch-independent pure logic that can be tested in isolation; src/context and src/engine depend on the ExecuTorch runtime. The interface directory implements the C-ABI call convention and is compiled into the shared library libphono_core.so for direct consumption by C code, Python bindings, mobile or other platforms.

## 项目结构

- src/core — 与 ExecuTorch 无关的纯逻辑，命名空间 phono::core
  - config — 加载模型包的 config.json 并校验维度配置；同时提供运行期 core_config 的解析与校验（解析 core_configs/default.json 中的运行参数，并对照模型硬限制返回错误枚举）
  - tokenizer — 汉字、上下文、拼音三份词表的编码与解码，拼音未命中时按编辑距离回退到最近音节
  - utf8_util — UTF-8 逐字符切分工具（基于 uni-algo）
- src/algo — 与 ExecuTorch 无关的解码算法，命名空间 phono::algo
  - zh2hans — 繁体转简体（zh2Hans）最长匹配替换，字典由 res/zh2hans.json 配置期 codegen
- src/context — 流式上下文状态，命名空间 phono::context
  - kv_cache — PersistentTensor：零拷贝的持久缓存缓冲区及其子视图
  - context — Context / ContextManager：B 路 self-KV Cache 视图、token id 序列，以及可复用上下文槽位管理
- src/engine — 推理引擎，命名空间 phono::engine
  - inference_engine — InferenceEngine 加载 v2 多方法 pre / post .pte 模块；InferenceSession 提供无状态 fill / generate API，以显式上下文槽位为参数
- src/custom_ops — ExecuTorch 自定义算子，命名空间 phono::ops：update_mhsa_kv
- interface — C-ABI 调用规范（phono_api.h / phono_api.cpp），编译为 libphono_core.so
- apps — 可执行程序：streaming_benchmark_demo_capi（C-ABI 基准演示）与 ime_demo_capi（交互式输入法演示）
- tests — 单元测试与模型级测试，由 CTest 驱动
- docs — 文档（docs/zh-cn 与 docs/en-us 分语言维护）
- third_party — ExecuTorch 源码，由 pixi 的 setup 任务拉取

## Project Layout

- src/core — ExecuTorch-independent pure logic, namespace phono::core
  - config — loads the model package's config.json and validates dimension configs; also parses and validates the runtime core_config (from core_configs/default.json) against the model's hard limits, returning an error enum instead of crashing
  - tokenizer — encode/decode for the Chinese, context and pinyin vocabularies; out-of-vocabulary pinyin falls back to the nearest syllable by edit distance
  - utf8_util — UTF-8 per-character splitting utilities (backed by uni-algo)
- src/algo — ExecuTorch-independent decoding algorithms, namespace phono::algo
  - zh2hans — Traditional->Simplified (zh2Hans) longest-match replacement; the table is code-generated from res/zh2hans.json at configure time
- src/context — streaming context state, namespace phono::context
  - kv_cache — PersistentTensor: zero-copy persistent cache buffers and sub-views
  - context — Context / ContextManager: B-wide self-KV-cache views, token-id sequences, and reusable context slots
- src/engine — inference engine, namespace phono::engine
  - inference_engine — InferenceEngine loads the v2 multi-method pre/post .pte modules; InferenceSession provides the stateless fill/generate APIs taking an explicit context slot
- src/custom_ops — ExecuTorch custom operator, namespace phono::ops: update_mhsa_kv
- interface — the C-ABI call convention (phono_api.h / phono_api.cpp), compiled into libphono_core.so
- apps — executables: streaming_benchmark_demo_capi (C-ABI benchmark) and ime_demo_capi (interactive IME demo)
- tests — unit and model-level tests driven by CTest
- docs — documentation, maintained separately under docs/zh-cn and docs/en-us
- third_party — ExecuTorch source, fetched by the pixi setup task

## 模型包

模型包是一个自包含的目录，InferenceEngine 以该目录路径构造：

- config.json — 模型与运行配置，含 common、pre_model、post_model、vocabs、decoding、runtime 六节
- bins/pre_model.pte — v2 多方法前段解码器（pre_model_pass1 / pre_model_pass2），维护 B 路 self-KV Cache
- bins/post_model.pte — 后段拼音编码器，输出 hidden states 与 logits mask
- vocabs/chinese_vocab.txt — 汉字词表（预测输出空间）
- vocabs/context_vocab.txt — 上下文词表（含特殊符号，如 bos_token）
- vocabs/pinyin_vocab.txt — 拼音音节词表（模型输入）
- dict/dict_trie.json — 词典 Trie，供 Viterbi 解码使用（可选）
- core_configs/default.json — 运行期 core_config：beam、slack_interval、min_accept_context、max_context_length、max_history_length、max_pinyin_length 与槽位淘汰参数

v2 模型可以通过 huggingface-cli 下载：

```
hf download afirelily/phonop2c_v2_0_alpha_05_base_model --local-dir ./phonop2c_v2_0_base_model
```

该 v2 模型要求 pre 程序包含 pre_model_pass1 与 pre_model_pass2 两个方法，并要求 post 方法名为 post_model；模型包的 config.json 中 model_format_version 为 2。请勿使用早期的 v1 模型包，v1 包缺少上述方法，无法被 v2 引擎加载。

## Model Package

A model package is a self-contained directory; InferenceEngine is constructed with its path:

- config.json — model and runtime config, in six sections: common, pre_model, post_model, vocabs, decoding, runtime
- bins/pre_model.pte — the v2 multi-method decoder (pre_model_pass1 / pre_model_pass2) with a B-wide self-KV cache
- bins/post_model.pte — the pinyin encoder, returning hidden states and a logits mask
- vocabs/chinese_vocab.txt — the Chinese-character vocabulary (prediction output space)
- vocabs/context_vocab.txt — the context vocabulary (including special tokens such as bos_token)
- vocabs/pinyin_vocab.txt — the pinyin-syllable vocabulary (model input)
- dict/dict_trie.json — the dictionary trie, used by Viterbi decoding (optional)
- core_configs/default.json — the runtime core_config: beam, slack_interval, min_accept_context, max_context_length, max_history_length, max_pinyin_length and slot-eviction parameters

Download the v2 model with huggingface-cli:

```
hf download afirelily/phonop2c_v2_0_alpha_05_base_model --local-dir ./phonop2c_v2_0_base_model
```

The v2 model requires the pre program to expose pre_model_pass1 and pre_model_pass2 and the post method to be named post_model; model_format_version is 2 in the package's config.json. Do not use earlier v1 packages — they lack the required methods and cannot be loaded by the v2 engine.

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

不同平台的 vcpkg 配置、CMake 生成器与编译工具链有所差异，请参阅 docs/zh-cn/build.md（简体中文）与 docs/en-us/build.md（English）中的分平台说明。其他文档（架构设计、core_config 参考、C-ABI 规范）同样按语言分别维护在 docs/zh-cn 与 docs/en-us 下。

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

Because the vcpkg configuration, the CMake generator and the compiler toolchain differ per platform, see docs/en-us/build.md (English) or docs/zh-cn/build.md (简体中文) for platform-specific instructions. The other documentation (architecture, core_config reference, C-ABI specification) is likewise maintained per language under docs/en-us and docs/zh-cn.

## 使用

下载模型之后，运行 C-ABI 演示程序：

```
results/streaming_benchmark_demo_capi phonop2c_v2_0_base_model
```

如需排除交互输入并进行可重复的性能测量，可向 CSV 基准程序传入模型包、采样次数和预热次数：

```
results/performance_benchmark phonop2c_v2_0_alpha_05_base_model 20 5
```

该程序报告模型加载耗时与 RSS、pre/post 各方法耗时、不同历史及拼音窗口长度下的生成耗时、单 token 增量 fill 耗时，以及 KV 清零、全量/增量 beam 重排和 session reset 的微基准。每个计时样本之前的输入构造、上下文填充和缓存初始化不计入样本耗时。

程序从 stdin 读取一行连续拼音窗口，例如 nihao，并通过 phono_tokenizer_separate_greedy 自动切分；可使用单引号显式提示边界，例如 ni'hao。每个窗口通过 InferenceSession::generate 执行 B 路 beam search；提交候选后下一次 fill 只对新增的严格因果历史做增量预填充。core_config 由程序从模型包的 core_configs/default.json 加载，也支持以第二个命令行参数传入自定义的 JSON 文件；当参数不符合模型限制（例如 beam 或上下文最大长度超出模型极限）时，程序打印错误枚举代码并退出。

模拟实时输入法编辑，可以运行：
```
results/ime_demo_capi phonop2c_v2_0_base_model
```
程序支持左右移动光标以及 Backspace/Delete 删除；拼音为空时，这些编辑键作用于已确认历史，模型仅接收历史光标之前的内容。此时还可用上下键切换历史，从而测试 ContextManager 的缓存复用；默认创建 2 个上下文，也可通过第三个参数指定数量。切分栏实时显示自动切分结果，下一栏显示当前历史，其余栏显示候选。输入候选编号即可在光标处提交到历史。按 Ctrl-C 退出。

## Usage

Run the C-ABI demo after downloading the model:

```
results/streaming_benchmark_demo_capi phonop2c_v2_0_base_model
```

For repeatable performance measurements without interactive I/O, run the CSV
benchmark with a model package, iteration count and warmup count:

```
results/performance_benchmark phonop2c_v2_0_alpha_05_base_model 20 5
```

It reports model-load RSS and latency, individual pre/post method latency,
generation across several history/window lengths, and one-token incremental
fill latency. Setup and cache initialization are outside each timed sample.

The program reads one unseparated pinyin window per line from stdin, e.g. nihao, and segments it with phono_tokenizer_separate_greedy; use a single quote to explicitly hint a boundary, e.g. ni'hao. Each window uses InferenceSession::generate for B-way beam search; after a candidate is committed, the next fill incrementally pre-fills only the new strictly-causal history. The core_config is loaded from the package's core_configs/default.json, and a custom JSON file can be passed as the second command-line argument; when a parameter does not fit the model (for example the beam or the context length exceeds a model limit), the program prints the error enum code and exits.

For real-time editing with left/right movement and backspace/delete, run:

```
results/ime_demo_capi phonop2c_v2_0_base_model
```

When pinyin is empty, those editing keys operate on committed history, and only the history prefix before the cursor is sent to the model. Up/Down then switches histories to exercise ContextManager cache reuse. Two contexts are created by default; pass a third argument to choose another count. The segmentation bar shows automatic segmentation, the next bar shows the active history, and the remaining bars show candidates after every edit. Type a candidate number to commit it at the cursor. Press Ctrl-C to exit.

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
