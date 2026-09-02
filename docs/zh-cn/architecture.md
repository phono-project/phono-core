# 架构设计

## 整体结构

推理引擎由两段模型组成：

- 前段模型：处理中文上下文，维护 B 路自注意力 KV 缓存，导出两个方法。pre_model_pass1 用于对严格因果历史做增量预填充；pre_model_pass2 用于在已填充的历史之上生成候选字符。缓存张量的形状为 [层数, 2, B, 前段最大长度, 头数, 头维度]。
- 后段模型：把拼音序列编码为隐藏状态与定宽候选 ID 表。运行时删除 padding 后只对真实候选执行 lm_head，并把稀疏列号映射回汉字词表 ID。输入长度受 config.json 中 post_model.max_seqlen 限制。

## 无状态会话与上下文槽位

InferenceSession 是无状态的：会话不保存任何对话状态，所有状态都存放在调用方显式传入的上下文槽位中。会话在构造时不绑定上下文，fill、replace_context、generate 等所有需要上下文的方法都以 context 作为参数传入。这样做有两个好处：单个会话可以驱动任意数量的上下文槽位，方便多会话复用与切换；同时 C 调用规范可以围绕「句柄 + 显式槽位」设计，避免把上下文生命周期耦合进会话内部。

每个上下文槽位包含三样东西：

- 前段自注意力 KV 缓存的视图（B 路），指向 ContextManager 的连续内存块；
- 已提交文本的上下文词表 id 序列（不含 BOS）；
- 两个游标：history_seqlen 表示已提交的严格因果历史长度（含 BOS），current_seqlen 表示缓存中下一个可写位置。

fill 与 replace_context 会改变 history_seqlen 与 current_seqlen；generate 只推进 current_seqlen，从不改变 history_seqlen，生成期间写入的缓存位置都是临时的，最终提交的候选通过 fill 固化到历史中。

## 增量填充

fill 对新增的严格因果历史做增量预填充：如果传入的 id 与当前历史存在公共前缀，只对新增区间运行 pre_model_pass1，并把 beam 0 的新切片复制到其余 beam。当历史超出 max_history_length 软上限时，会先做左移窗口：丢弃最旧的历史 token，把已提交历史截断到 max_history_length - slack_interval，同时保留第一个 BOS 作为注意力汇点，然后再追加新内容。replace_context 处理三种情况：完全一致的快照直接复用；历史是目标的公共前缀时增量填充；目标发生分歧时从 BOS 重新填充。

## 生成与上下文窗口

generate 的流程如下：

1. 若调用方提供了完整上下文，先执行 replace_context 把目标上下文固化到槽位；
2. 若历史长度达到 max_history_length，先截断到 max_history_length - slack_interval（保留 BOS）；
3. 校验拼音窗口长度不超过 max_pinyin_length，并校验生成位置不会越过前段模型的硬上限；
4. 运行后段模型编码拼音，再按 B 路 beam search 逐字生成，每一步读取严格因果历史与后段的交叉注意力；
5. 返回各 beam 的得分、候选 id 与解码文本。

其中 max_history_length 与 max_pinyin_length 的取值必须满足约束 max_history_length < max_context_length - max_pinyin_length，否则在已提交文本到达前段缓存上限之前就可能发生溢出。该约束在 core_config 解析阶段强制校验，违反时返回错误枚举。

## 中止语义与游标回滚

generate 通过「回调函数 + 上下文指针」检查中止状态：调用方传入一个返回布尔值的回调与一个不透明的上下文指针，generate 在每一步之间轮询回调。回调返回真时，generate 返回 Cancelled 错误，并回滚生成游标，把 current_seqlen 恢复到 history_seqlen，使上下文回到「只含已提交历史」的一致状态。中断不会回滚整个 KV 缓存的内容，生成位置残留的临时状态会在下一次 generate 或 fill 时被覆盖。这种设计保证了中断后的 generate 可以从已提交历史重新开始，而无需清理缓存。

## 上下文槽位管理

ContextManager 维护一组上下文槽位，每个槽位持有前段自注意力 KV 缓存与上下文 id 序列的连续内存视图。槽位按代价评估进行淘汰与复用：前缀复用时直接截断多余尾部；BOS 保护的左移窗口复用时丢弃最旧 token 并保留匹配后缀；找不到可复用匹配时选择代价最弱的槽位重算。匹配后缀的最小长度由 min_accept_context 控制，窗口复用时希望留出的松弛量由 slack_interval 控制，槽位的价值随使用频率衰减，由 trial_ratio、decay_alpha、decay_lambda 控制。
