# C-ABI 调用规范

共享库 libphono_core.so（Windows 下为 phono_core.dll，macOS 下为 libphono_core.dylib）导出稳定的 C 调用规范，声明位于 interface/phono_api.h。该接口使用不透明句柄，所有 C++ 异常都不会越过 ABI 边界，失败一律以错误枚举代码返回。

## 调用流程

典型的调用顺序如下：

1. phono_engine_create 加载模型包目录，得到引擎句柄；
2. phono_context_manager_create 传入 core_config 的 JSON 字符串与槽位数量，得到上下文管理器句柄；
3. phono_context_manager_get_auto 或 phono_context_manager_get_by_id 取得上下文槽位句柄；
4. phono_session_create 传入同一份 core_config 的 JSON 字符串，得到会话句柄（会话不绑定槽位）；
5. 循环调用 phono_tokenizer_separate_greedy 切分原始拼音、phono_tokenizer_encode_pinyin 编码拼音、phono_session_generate 生成候选、phono_generate_result_free 释放结果、phono_session_fill 提交候选；
6. 结束后依次释放会话、管理器与引擎。

## 错误码

| 枚举 | 含义 |
| --- | --- |
| PHONO_OK | 成功 |
| PHONO_INVALID_ARGUMENT | 参数非法（空指针、负长度、id 越界等） |
| PHONO_CONTEXT_LIMIT_EXCEEDED | 上下文超过前段模型硬上限 |
| PHONO_PINYIN_LIMIT_EXCEEDED | 拼音窗口超过 max_pinyin_length 或后段硬限制 |
| PHONO_NO_CANDIDATES | 候选不足，无法填满 beam |
| PHONO_CANCELLED | 回调要求中止，游标已回滚 |
| PHONO_MODEL_ERROR | 模型加载或运行失败 |
| PHONO_CONFIG_ERROR | core_config 解析失败或参数超出模型限制 |

phono_error_name 返回错误码的稳定名称；phono_last_error_message 返回最近一次失败的线程局部诊断信息。

## core_config 传参

上下文管理器与会话的创建函数都以 JSON 字符串接收 core_config，例如：

```c
const char* config_json =
    "{\"beam_size\":3,\"slack_interval\":8,\"min_accept_context\":8,"
    "\"max_context_length\":127,\"max_history_length\":94,\"max_pinyin_length\":32}";
phono_context_manager* manager = NULL;
if (phono_context_manager_create(engine, config_json, 1, &manager) != PHONO_OK) {
    fprintf(stderr, "%s\n", phono_last_error_message());
}
```

库内部用 nlohmann-json 反序列化并对照模型限制校验，参数不满足条件（例如 beam 或上下文最大长度超出模型极限）时返回 PHONO_CONFIG_ERROR。调用方无需也无法自行拼装模型维度的内部结构，直接传 JSON 字符串即可。

## 会话与槽位

会话是无状态的，创建时不绑定任何槽位，所有方法都显式接收 phono_context 句柄，因此可以驱动任意槽位。generate 的拼音窗口受 core_config 的 max_pinyin_length 约束，上下文受 max_history_length 约束；context_ids 为空表示复用槽位中已提交的历史。中止通过回调完成，回调在每步之间被轮询，返回非零即中止并回滚游标：

```c
int my_cancel(void* user_data) {
    (void)user_data;
    return g_stop_flag;   /* 由信号或其它线程置位 */
}
```

## 内存所有权

- 编码接口（phono_tokenizer_encode_context、phono_tokenizer_encode_pinyin）输出的 id 数组由调用方负责用 phono_free 释放；
- phono_tokenizer_separate_greedy 返回的音节数组是单块连续内存，调用一次 phono_free 即可释放；
- phono_tokenizer_decode 返回的字符串由调用方用 phono_free 释放；
- phono_session_generate 填充的 phono_generate_result 内部（beams、pred_ids、decoded）由 phono_generate_result_free 整体释放；
- 句柄分别由对应的 destroy 函数释放：phono_engine_destroy、phono_context_manager_destroy、phono_session_destroy。

## 完整函数列表

见 interface/phono_api.h 中的声明与注释，包括：phono_engine_create/destroy、phono_engine_pre_max_seqlen、phono_engine_post_max_seqlen、phono_engine_beam_size、phono_tokenizer_separate_greedy、phono_tokenizer_encode_context、phono_tokenizer_encode_pinyin、phono_tokenizer_decode、phono_tokenizer_chinese_to_context、phono_context_manager_create/destroy/num_contexts/get_by_id/get_auto、phono_context_ids_len、phono_context_current_seqlen、phono_context_history_seqlen、phono_session_create/destroy/beam_size/reset/fill/replace_context/generate、phono_generate_result_free、phono_free、phono_error_name、phono_last_error_message。

## 演示程序

streaming_benchmark_demo_capi 是使用该 C 调用规范的完整示例：自动切分一行原始拼音，生成候选并选择提交。它的源码位于 apps/streaming_benchmark_demo_capi.cpp。
