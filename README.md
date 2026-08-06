# phono-core

## 简介

phono-core 是 PhonoP2C 的边缘侧 C++ 推理引擎：在 ExecuTorch 运行时上部署两段式 PostfixLM 拼音转汉字模型，输入「中文上下文 + 拼音音节」，输出汉字候选。它只依赖 ExecuTorch 的 C++ API 与少量系统库，适合部署到移动端或嵌入式设备。

代码按功能划分为 src/core、src/algo、src/context、src/engine 四个子目录，对应 phono::core、phono::algo、phono::context、phono::engine 四个子命名空间，另有一个 src/custom_ops 目录（命名空间 phono::ops）提供 ExecuTorch 自定义算子。其中 src/core 与 src/algo 不依赖 ExecuTorch，是纯逻辑层，可独立测试；src/context 与 src/engine 依赖 ExecuTorch 运行时。

## Introduction

phono-core is the on-device C++ inference engine for PhonoP2C: it deploys the two-stage PostfixLM pinyin-to-Chinese model on the ExecuTorch runtime, taking "Chinese context + pinyin syllables" as input and producing Chinese-character candidates. It depends only on ExecuTorch's C++ API and a few system libraries, making it suitable for mobile or embedded deployment.

The code is organized into four subdirectories — src/core, src/algo, src/context, src/engine — mirrored by the sub-namespaces phono::core, phono::algo, phono::context and phono::engine, plus src/custom_ops (namespace phono::ops) for ExecuTorch custom operators. src/core and src/algo are ExecuTorch-independent pure logic that can be tested in isolation; src/context and src/engine depend on the ExecuTorch runtime.

## 项目结构

- src/core — 与 ExecuTorch 无关的纯逻辑，命名空间 phono::core
  - config — 加载模型包的 config.json 并校验维度配置
  - tokenizer — 汉字、上下文、拼音三份词表的编码与解码，拼音未命中时按编辑距离回退到最近音节
  - utf8_util — UTF-8 逐字符切分工具
- src/algo — 与 ExecuTorch 无关的解码算法，命名空间 phono::algo
  - trie — 词典 Trie 的加载与词匹配，产出 WordMatch 候选
  - viterbi — 词典约束的 Viterbi N-best 束搜索解码
- src/context — 流式上下文状态，命名空间 phono::context
  - kv_cache — PersistentTensor：零拷贝的持久缓存缓冲区及其子视图
  - context — Context / ContextManager：单个会话的 KV Cache 视图、token id 序列与游标，以及多上下文管理
- src/engine — 推理引擎，命名空间 phono::engine
  - inference_engine — InferenceEngine 加载 pre / post 两个 .pte 模块，InferenceSession 提供流式预测 API
- src/custom_ops — ExecuTorch 自定义算子，命名空间 phono::ops：update_kv_cache / update_cross_kv / update_mhsa_kv
- apps — 可执行程序：streaming_benchmark_demo 交互式流式基准
- third_party — ExecuTorch 源码，由 pixi 的 setup 任务拉取

## Project Layout

- src/core — ExecuTorch-independent pure logic, namespace phono::core
  - config — loads the model package's config.json and validates dimension configs
  - tokenizer — encode/decode for the Chinese, context and pinyin vocabularies; out-of-vocabulary pinyin falls back to the nearest syllable by edit distance
  - utf8_util — UTF-8 per-character splitting utilities
- src/algo — ExecuTorch-independent decoding algorithms, namespace phono::algo
  - trie — dictionary-trie loading and word matching, producing WordMatch candidates
  - viterbi — dictionary-constrained Viterbi N-best beam-search decoding
- src/context — streaming context state, namespace phono::context
  - kv_cache — PersistentTensor: zero-copy persistent cache buffers and sub-views
  - context — Context / ContextManager: per-conversation KV-cache views, token-id sequence and cursor, plus multi-context management
- src/engine — inference engine, namespace phono::engine
  - inference_engine — InferenceEngine loads the pre/post .pte modules; InferenceSession provides the streaming prediction API
- src/custom_ops — ExecuTorch custom operators, namespace phono::ops: update_kv_cache / update_cross_kv / update_mhsa_kv
- apps — executables: streaming_benchmark_demo, an interactive streaming benchmark
- third_party — ExecuTorch source, fetched by the pixi setup task

## 模型包

模型包是一个自包含的目录，InferenceEngine 以该目录路径构造：

- config.json — 模型与运行配置，含 common、pre_model、post_model、vocabs、decoding、runtime 六节
- bins/pre_model.pte — 前段因果编码器：编码上下文并维护 KV Cache
- bins/post_model.pte — 后段双向解码器：读取拼音音节，交叉注意力到上下文，输出汉字分布
- vocabs/chinese_vocab.txt — 汉字词表（预测输出空间）
- vocabs/context_vocab.txt — 上下文词表（含特殊符号，如 bos_token）
- vocabs/pinyin_vocab.txt — 拼音音节词表（模型输入）
- dict/dict_trie.json — 词典 Trie，供 Viterbi 解码使用

示例使用的模型可以通过 huggingface-cli 下载：
```
hf download afirelily/phonop2c_v1_0_base_model --local-dir ./phonop2c_v1_0_base_model
```

## Model Package

A model package is a self-contained directory; InferenceEngine is constructed with its path:

- config.json — model and runtime config, in six sections: common, pre_model, post_model, vocabs, decoding, runtime
- bins/pre_model.pte — the causal pre-model: encodes the context and maintains the KV caches
- bins/post_model.pte — the bidirectional post-model: reads pinyin syllables, cross-attends to the context, and outputs the Chinese-character distribution
- vocabs/chinese_vocab.txt — the Chinese-character vocabulary (prediction output space)
- vocabs/context_vocab.txt — the context vocabulary (including special tokens such as bos_token)
- vocabs/pinyin_vocab.txt — the pinyin-syllable vocabulary (model input)
- dict/dict_trie.json — the dictionary trie, used by Viterbi decoding

The sample model package can be downloaded using huggingface-cli:
```
hf download afirelily/phonop2c_v1_0_base_model --local-dir ./phonop2c_v1_0_base_model
```


## 构建与运行

前置要求：pixi 环境。ExecuTorch 由 pixi 任务拉取源码后随主工程一起编译：

- pixi run setup — 将 ExecuTorch（含子模块）克隆到 third_party/executorch，已存在时跳过
- pixi run config — 配置 CMake 构建
- pixi run build — 增量编译

配置阶段可在 CMake 缓存中覆写两个选项：PHONO_USE_INSTALLED_EXECUTORCH 指定使用独立安装的 ExecuTorch（需同时设置 CMAKE_PREFIX_PATH）；EXECUTORCH_SOURCE_DIR 指定 ExecuTorch 源码路径（默认 third_party/executorch）。可执行文件默认输出到 results/ 目录。

## Build & Run

Prerequisites: the pixi environment. The pixi task fetches ExecuTorch's source and it is compiled together with this project:

- pixi run setup — clones ExecuTorch (with submodules) into third_party/executorch; skipped if already present
- pixi run config — configures the CMake build
- pixi run build — incremental build

At configure time, two CMake cache variables can be overridden: PHONO_USE_INSTALLED_EXECUTORCH selects a separately installed ExecuTorch (set CMAKE_PREFIX_PATH accordingly), and EXECUTORCH_SOURCE_DIR points to the ExecuTorch source (default third_party/executorch). Executables are output to results/ by default.

## 使用

下载模型之后，运行基准 demo：
```
results/streaming_benchmark_demo phonop2c_v1_0_base_model
```

程序从 stdin 读取一行空格分隔的拼音窗口，例如 ni hao。每个窗口的处理分两步：先把自上次提交以来新增的已定稿文本增量编码进上下文（pre 模型），再对当前拼音窗口执行解码（post 模型），随后打印 Viterbi N-best 候选及得分。交互选择一个候选序号提交到上下文；当上下文触及 pre 模型的序列上限，或按 Ctrl-C 时结束，并打印基准报告，包括窗口数、提交字符数、每步 Viterbi 延迟与吞吐。

## Usage

Run the benchmark demo after downloading the model:
```
results/streaming_benchmark_demo phonop2c_v1_0_base_model
```

The program reads one space-separated pinyin window per line from stdin, e.g. ni hao. Each window is processed in two steps: first the text committed since the last call is incrementally encoded into the context (pre model), then the current pinyin window is decoded (post model), and the Viterbi N-best candidates with scores are printed. Pick a candidate number interactively to commit it to the context; the run ends when the context reaches the pre-model sequence limit or on Ctrl-C, printing a benchmark report with window count, committed characters, per-step Viterbi latency and throughput.

## 流式推理设计

InferenceSession 是无状态的：所有会话状态都保存在传入的 Context 中，因此一个 session 可以驱动多个上下文。每个 Context 持有三样东西：KV Cache 的 PersistentTensor 视图、已提交文本的 token id 序列、以及游标 current_position（即下一次写入的缓存位置）。

- advance_context 只编码自上次调用以来的增量文本，经 pre 模型写入 KV Cache 的新区间，旧区间原样保留
- ContextManager 把多个上下文的 KV Cache 堆叠为单块连续分配，每个上下文只是其中的一个视图，避免逐上下文分配与内存碎片化
- src/custom_ops 中的三个算子把新计算的 K / V 张量写入缓存指定位置，供导出的 .pte 图直接调用

## Streaming Inference Design

InferenceSession is stateless: all per-conversation state lives in the Context passed in, so one session can drive many contexts. Each Context holds three things: PersistentTensor views into the KV caches, the token-id sequence of the committed text, and the cursor current_position (the cache write position for the next step).

- advance_context encodes only the incremental text committed since the last call, runs it through the pre model into the new KV-cache region, and leaves older regions untouched
- ContextManager stacks all contexts' KV caches into a few contiguous allocations; each context is just a view into them, avoiding per-context allocation and memory fragmentation
- the three operators in src/custom_ops write freshly computed K / V tensors into the cache at the given position, and are called directly by the exported .pte graphs

## 许可证与最终声明

由于本人精力有限，对架构、标准的设计无法全力，而且文档大部分使用了 LLM 生成，不可避免的会存在纰漏、更新不及时等问题。如果发现有任何问题，欢迎提出 issues。

本项目基于 Apache License 2.0 开源，详见 [LICENSE](LICENSE)。

## License and Final Note

Due to my limited time and resources, I am unable to devote my full attention to architectural and standard design. Furthermore, since most of the documentation was generated by an LLM, there will inevitably be some inaccuracies and delays in updates. If you find any issues, please feel free to open an issue.

Open-sourced under the Apache License 2.0; see [LICENSE](LICENSE).
