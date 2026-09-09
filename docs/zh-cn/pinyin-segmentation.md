# 智能拼音分词

## Scorer 与 MAP 解码

分词模型只负责给字符间隙打分，拼音词表 Trie 负责约束合法性。长度为 `n` 的归一化输入恒有 `n-1` 个间隙。令模型在第 `k` 个间隙输出 logit `z_k`，边界状态为 `y_k`，则

```text
log P(Y|X) = sum_k [y_k log sigmoid(z_k) + (1-y_k) log(1-sigmoid(z_k))]
           = C(X) + sum_k y_k z_k.
```

`C(X)` 与候选路径无关。因此只求 MAP 最优路径时，只需累加被选为边界的原始 logits，无需 sigmoid、几何均值或音节数量惩罚。这个化简只保持 argmax，不代表输出了校准后的整条路径概率。

Trie 从每个字符位置枚举所有全拼与简拼词条，形成前向 DAG。边 `(i,j)` 在 `j<n` 时获得 `z_(j-1)`，到达结尾时权重为零。动态规划在每个字符偏移保留最优前缀，复杂度为 `O(E)`。分数完全相同时依次偏好更少音节、较早位置更长的音节。

## 自动路由

`InferenceEngine::segment_pinyin` 与 C ABI 的 `phono_engine_segment_pinyin` 是稳定高层入口：

- 模型包含 `segmenter` 且归一化长度不小于 `min_input_chars` 时，使用 Scorer + Trie DAG；
- 输入长度为 1 或 2 时不调用模型，直接使用 checked FMM；
- 模型包没有 `segmenter` 时，引擎仍可加载，所有输入使用 checked FMM，并在 `phono_engine_info_json` 的 `diagnostics` 中报告警告；
- 模型包声明了 `segmenter`、但文件、词表或方法损坏时，引擎加载失败，不静默回退。

checked FMM 使用全局可达性检查：先最小化非法字符数，再按最大匹配规则打破平局，不会因局部最长边而走进本可避免的死路。

## 归一化和非法输入

归一化按 `engine_config.json` 执行：ASCII 大写转小写；配置的分隔符被删除并强制产生边界；`j/q/x/y` 后且未跨越强制边界的 `v` 映射为 `u`。因此 `jvan` 变为 `juan`，而 `lv`、`nv` 保留原状。每一步修改都出现在 `normalization_events` 中。

严格模式只允许合法 Trie 边。不存在完整合法路径时返回 `InferenceError::InvalidPinyin` / `PHONO_INVALID_PINYIN`，且不会执行 Scorer。

安全模式保留逐字符非法边参与 DP，但优化顺序首先最小化非法字符数量，所以只有不存在全合法路径时才会选中非法边：

- `repair=false`（默认）：删除选中的非法字符；
- `repair=true`：每个非法字符分别通过 `find_pinyin_id_nearest` 修复为合法 token；编辑距离不作为公共接口返回。

`invalid_ranges` 始终采用归一化前原始 UTF-8 字符串的半开字节坐标 `[begin,end)`。安全模式成功时，`normalized_input` 和 `segments` 保证可由拼音 tokenizer 精确编码；若删除后为空，仍返回 `PHONO_INVALID_PINYIN`。

## 版本化 JSON Schema 1.0

所有 C ABI JSON 文档都要求顶层 `schema_version` 为字符串 `"1.0"`；缺失或未知版本返回 `PHONO_CONFIG_ERROR`。

分词请求：

```json
{"schema_version":"1.0","input":"jv'anpv"}
```

分词响应（即使函数返回 `PHONO_INVALID_PINYIN`，只要分配成功仍会提供响应 JSON）：

```json
{
  "schema_version": "1.0",
  "status": {"ok": true, "code": "ok"},
  "input": "jv'anpv",
  "canonical_input": "juanpv",
  "normalized_input": "juanp",
  "strategy": "scorer_viterbi",
  "segments": ["juan", "p"],
  "pinyin_ids": [167, 9],
  "invalid_ranges": [
    {"begin": 6, "end": 7, "text": "v", "replacement": ""}
  ],
  "normalization_events": [
    {"type":"v_to_u","begin":1,"end":2,"before":"v","after":"u"},
    {"type":"forced_boundary","begin":2,"end":3,"before":"'","after":""},
    {"type":"invalid_deleted","begin":6,"end":7,"before":"v","after":""}
  ]
}
```

字段约束：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| status | object | `ok` 与稳定错误名 `code` |
| canonical_input | string | 字符归一化和分隔符删除后、非法处理前的 Scorer 输入 |
| normalized_input | string | 成功时绝对合法的最终拼音串 |
| strategy | string | `scorer_viterbi` 或 `checked_fmm` |
| segments | string[] | 最终合法 token；与 `pinyin_ids` 一一对应 |
| pinyin_ids | int32[] | 可直接传给 `phono_session_generate` |
| invalid_ranges | object[] | 原始输入字节范围、原文及修复串；空修复串表示删除 |
| normalization_events | object[] | 按执行顺序列出的全部规范化、删除或修复事件 |

引擎信息由 `phono_engine_info_json` 返回：

```json
{
  "schema_version":"1.0",
  "model_version":"v2_2-base-alpha05-w8a8",
  "model_format_version":"2.2",
  "segmenter":{"available":true,"fallback":null},
  "diagnostics":[]
}
```

模型包 `config.json` 的可选配置如下。`layout` 固定为 `BHWC`；`min_input_chars` 至少为 3：

```json
"segmenter": {
  "model_path": "bins/pinyin_segment.pte",
  "method": "forward",
  "char_vocab": "vocabs/pinyin_char_vocab.txt",
  "min_input_chars": 3,
  "max_input_chars": 512,
  "layout": "BHWC",
  "quantization": "none"
}
```
