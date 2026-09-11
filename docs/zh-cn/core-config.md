# 运行期配置参考

运行策略与模型包分离，并拆成两份版本化 JSON。每份文档都必须包含字符串 `"schema_version": "1.0"`；缺失或版本未知时返回 `PHONO_CONFIG_ERROR`。

## 引擎与 tokenizer 策略

`core_configs/engine_config.json` 传给 `phono_engine_create`：

```json
{
  "schema_version": "1.0",
  "tokenizer": {
    "normalization": {
      "lowercase_ascii": true,
      "normalize_v_to_u": true,
      "separators": "'"
    },
    "segment_mode": "safe",
    "repair": false,
    "max_pinyin_chars": 128
  }
}
```

| 字段 | 含义 |
| --- | --- |
| `lowercase_ascii` | 匹配前把 ASCII 大写字母转为小写 |
| `normalize_v_to_u` | 只在同一分隔片段的 `j/q/x/y` 后把 `v` 转为 `u` |
| `separators` | 删除这些字符，并在原位置强制产生音节边界 |
| `segment_mode` | `safe` 返回合法输出和非法范围；`strict` 拒绝任何非法路径 |
| `repair` | 安全模式下逐字符映射到最近合法 token，而不是删除非法字符 |
| `max_pinyin_chars` | 分词接口允许的最大归一化字符数 |

`max_pinyin_chars` 限制分词工作量；`max_pinyin_length` 则限制送入 P2C 模型的音节 ID 数量，两者含义不同。路由、归一化和响应 Schema 见 [智能拼音分词](pinyin-segmentation.md)。

## 上下文与会话策略

`core_configs/context_manager.json` 分别传给 `phono_context_manager_create` 和 `phono_session_create`：

```json
{
  "schema_version": "1.0",
  "beam_size": 3,
  "slack_interval": 8,
  "min_accept_context": 8,
  "max_context_length": 127,
  "max_history_length": 94,
  "max_pinyin_length": 32,
  "trial_ratio": 0.25,
  "decay_alpha": 0.5,
  "decay_lambda": 0.02
}
```

| 字段 | 约束 | 缺省值 |
| --- | --- | --- |
| `beam_size` | 等于 pre pass-2 的 batch size | 模型 batch size |
| `slack_interval` | `0 <= value < max_history_length` | 最大为 8 |
| `min_accept_context` | 大于零 | 8 |
| `max_context_length` | 不超过 `pre_model.max_seqlen - 1` | 该上限 |
| `max_history_length` | 小于 `max_context_length - max_pinyin_length` | 最大合法值 |
| `max_pinyin_length` | `1..post_model.max_seqlen` | post 模型上限 |
| `trial_ratio` | `(0,1]` | 0.25 |
| `decay_alpha` | 非负 | 0.5 |
| `decay_lambda` | 非负 | 1/60 |

除 `schema_version` 外的字段可以省略并使用模型推导值。`N`、`T` 等旧键名不再生效。

历史窗口采用保留 BOS 的左移机制：到达 `max_history_length` 时丢弃最旧 token，缩到 `max_history_length - slack_interval`。超出模型限制时，C++ 返回 `CoreConfigError`，C ABI 返回 `PHONO_CONFIG_ERROR`，详细原因可通过 `phono_last_error_message` 获取。
