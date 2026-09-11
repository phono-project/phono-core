# C ABI 调用规范

`libphono_core` 导出 `interface/phono_api.h` 中声明的异常安全 C ABI。句柄均为不透明类型，所有失败都表示为 `phono_status`；`phono_last_error_message()` 提供线程局部的诊断文本。

## 典型调用流程

1. 用 `phono_engine_create` 加载版本化 `engine_config.json`。
2. 用 `phono_engine_info_json` 一次性检查分词模型可用性和加载诊断。
3. 把原始拼音交给 `phono_engine_segment_pinyin`；响应已经包含合法的 `pinyin_ids`。
4. 使用版本化 `context_manager.json` 创建上下文管理器和会话，再调用 `phono_session_generate`。
5. 用 `phono_session_fill` 提交候选并继续下一轮。

模型包内有可用 segmenter 时，分词接口自动使用 Scorer + Viterbi，否则使用 checked FMM。完整请求和响应 Schema 见[智能拼音分词](pinyin-segmentation.md)。

```c
phono_engine* engine = NULL;
phono_status status = phono_engine_create(model_dir, engine_config_json, &engine);
if (status != PHONO_OK) {
    fprintf(stderr, "%s: %s\n", phono_error_name(status),
            phono_last_error_message());
}

char* result_json = NULL;
status = phono_engine_segment_pinyin(
    engine, "{\"schema_version\":\"1.0\",\"input\":\"nihaoma\"}",
    &result_json);
/* 解析 pinyin_ids 和 invalid_ranges，然后释放 JSON。 */
phono_free(result_json);
```

## 状态码

| 状态 | 含义 |
| --- | --- |
| `PHONO_OK` | 成功 |
| `PHONO_INVALID_ARGUMENT` | 空指针、负数量、非法 ID 等调用错误 |
| `PHONO_CONTEXT_LIMIT_EXCEEDED` | 上下文超过 pre 模型硬上限 |
| `PHONO_PINYIN_LIMIT_EXCEEDED` | 字符或音节窗口超过配置上限 |
| `PHONO_INVALID_PINYIN` | 严格模式不存在合法路径，或安全删除后输入为空 |
| `PHONO_NO_CANDIDATES` | 生成过程没有合法候选 |
| `PHONO_CANCELLED` | 回调中止生成，游标已回滚 |
| `PHONO_MODEL_ERROR` | 模型包加载或模型执行失败 |
| `PHONO_CONFIG_ERROR` | 配置、请求或 Schema 版本无效 |

`phono_error_name` 返回稳定的符号名。

## 配置与会话

引擎/tokenizer 策略与上下文/会话策略是两份不同的 JSON，见 [运行期配置](core-config.md)。会话不绑定上下文槽位，每个操作都显式接收 `phono_context`，因此一个会话可以驱动多个槽位。生成时不传 `context_ids` 表示复用槽位中的已提交历史。取消回调返回非零时在步骤之间中止，并把 `current_seqlen` 恢复为 `history_seqlen`。

## 内存所有权

- `phono_engine_info_json`、`phono_engine_segment_pinyin` 和 `phono_tokenizer_decode` 返回的字符串用 `phono_free` 释放。
- `phono_tokenizer_encode_context` 返回的数组用 `phono_free` 释放。
- `phono_generate_result` 内部资源用 `phono_generate_result_free` 释放。
- engine、manager、session 句柄用各自的 destroy 函数释放。
- context 句柄从 manager 借用，调用方不得释放。

完整函数列表以 `interface/phono_api.h` 为准。`cli_demo_capi` 与 `ime_demo_capi` 是端到端 C ABI 示例，都会展示选中的分词路由和非法范围。
