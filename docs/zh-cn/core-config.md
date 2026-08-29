# core_config 参考

core_config 是运行期配置，与模型包分离。它由 JSON 对象表示，演示程序从模型包的 core_configs/default.json 读取，也支持以命令行参数传入自定义 JSON；C 调用规范则以 JSON 字符串的形式把整个 core_config 交给共享库，由库内部解析与校验。

配置会在解析阶段对照模型包的硬限制进行校验，参数不满足条件时返回错误枚举代码（C++ 层为 CoreConfigError，C 层为 PHONO_CONFIG_ERROR），而不是抛出异常或静默回退。

## 配置项

| 键 | 说明 | 约束 | 默认值（随模型推导） |
| --- | --- | --- | --- |
| beam_size | 束搜索宽度，必须等于模型的 beam 批宽度 | 等于前段 pass2 的批大小 | 模型批大小 |
| slack_interval | 窗口松弛区间，触发窗口截断时留出的余量 | 0 ≤ slack_interval < max_history_length | 8 |
| min_accept_context | 槽位复用要求的最短可复用后缀长度 | > 0 | 8 |
| max_context_length | 每个槽位可容纳的已提交上下文 id 数量 | ≤ pre_model.max_seqlen - 1 | pre_model.max_seqlen - 1 |
| max_history_length | 已提交历史的软上限，达到后先截断再生成 | 必须满足 max_history_length < max_context_length - max_pinyin_length | max_context_length - max_pinyin_length - 1 |
| max_pinyin_length | 拼音窗口长度上限 | ≥ 1 且 ≤ post_model.max_seqlen（后段硬限制） | post_model.max_seqlen |
| trial_ratio | 槽位淘汰时允许进入试用池的比例 | 0 < trial_ratio ≤ 1 | 0.25 |
| decay_alpha | 槽位价值随长度增长的指数 | ≥ 0 | 0.5 |
| decay_lambda | 槽位价值随时间衰减的速率 | ≥ 0 | 1/60 |

未出现的键会使用「随模型推导的默认值」，因此一个空对象也能解析出对当前模型合法的一组配置。旧版使用的键名 N 与 T 已被 slack_interval 与 min_accept_context 取代，N 与 T 不再生效。

## 随模型推导的默认值

对于模型包 config.json 中的维度，默认值按以下规则推导：

- beam_size 取 runtime.batch_size（亦即前段 pass2 的批大小）；
- max_context_length 取 pre_model.max_seqlen - 1；
- max_pinyin_length 取 post_model.max_seqlen；
- max_history_length 取 max_context_length - max_pinyin_length - 1，保证严格小于 max_context_length - max_pinyin_length；
- slack_interval 取 min(8, max_history_length - 1)。

仓库内附带的 core_configs/default.json 是针对 v2_0_alpha_05 模型（前段 max_seqlen 128，后段 max_seqlen 32，批大小 3）的取值：beam_size 为 3，slack_interval 为 8，min_accept_context 为 8，max_context_length 为 127，max_history_length 为 94，max_pinyin_length 为 32。

## 窗口机制

前段模型对序列长度存在硬上限 pre_model.max_seqlen，后段模型对拼音输入长度存在硬上限 post_model.max_seqlen。为避免上下文超长或状态陷入混乱，运行期通过两个上限管理窗口：

- 拼音窗口：generate 输入的拼音音节数不得超过 max_pinyin_length，同时不得超过后段模型的硬限制 post_model.max_seqlen。超出时返回 PinyinLimitExceeded。
- 历史窗口：已提交历史不得超过软上限 max_history_length。fill 追加历史时若会超过该上限，先把历史截断到 max_history_length - slack_interval，保留第一个 BOS 作为注意力汇点，再追加新内容；generate 开始前若历史已达到 max_history_length，同样先截断到 max_history_length - slack_interval。

截断采用左移窗口：丢弃最旧的 token，缓存内容按 token 粒度整体前移，BOS 始终占据位置 0。这样既避免上下文无限增长，也保留了尽可能多的近期上下文。

## 一致性约束

以下约束在解析时强制校验，违反时分别返回对应错误：

- beam_size 必须等于模型 beam 批宽度，否则返回 BeamSizeMismatch；
- max_context_length + 1 必须不超过 pre_model.max_seqlen，否则返回 MaxContextLengthExceeded；
- max_pinyin_length 不得大于 post_model.max_seqlen，否则返回 MaxPinyinLengthInvalid；
- max_history_length 必须严格小于 max_context_length - max_pinyin_length，否则返回 MaxHistoryLengthInvalid。该约束保证即使历史与拼音窗口同时取到上限，已提交文本也不会在下次上屏之前溢出前段缓存；
- slack_interval 必须在 [0, max_history_length) 区间内，否则返回 SlackInvalid；
- min_accept_context 必须大于 0，否则返回 MinAcceptContextInvalid。

C++ 层错误枚举为 core::CoreConfigError，可通过 core_config_error_name 转成可读名称；C 层统一映射为 PHONO_CONFIG_ERROR，详细原因见 phono_last_error_message 的说明。
