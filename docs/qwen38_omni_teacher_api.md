# 教师 API 验证

当前适配器使用 qwen3.8-omni-flash，通过阿里云 Chat Completions 接口复用同一个三阶段 Agent。它没有独立 Planner，Memory 使用与本地模型相同的观察记录格式。

## 凭据与输入

复制 .env.example 为 .env，在本机填入 DASHSCOPE_API_KEY 和匹配地域的 DASHSCOPE_BASE_URL；.env 不得提交。代码仅向 aliyuncs.com 官方 HTTPS 域名发送凭据。

```bash
bash scripts/test_api_teacher.sh --manifest /path/to/manifest.jsonl --sample-id YOUR_SAMPLE_ID --max-steps 64 --allow-external --output runs/api_teacher_check
```

--allow-external 表示确认所选访谈允许发送给服务商。--prepare-only 只生成本地计划，不调用 API。运行使用专用 Conda 环境和 FFmpeg。

Manifest 输入包括 id / dataset / split / person_id / video / segments；人物级 RecruitView 还保留各问答的 question 和拼接时间。脚本只向策略传递白名单原始字段，不传人格标签、旧标注或已有模型答案。正式训练轨迹必须来自 Train，Dev 诊断不得直接进入 SFT。

## 流程与协议

- Stage 1 按来源生成 summary 和可为空的 candidates。Python 绑定真实来源窗口，不由模型生成来源 ID。
- Select 由程序先调度粗扫候选，再按入池顺序调度细化；没有模型 STOP 动作。
- Observe 返回 observations 和可为空的 candidates。local_time 使用当前片段从零播放的秒数；历史覆盖也使用同一局部坐标。未验证 ASR 不提供精确媒体位置。
- Memory 保存完整原文与 M 引用。读取时只把 observations 当作已观察内容，未来计划不进入评分或 Final 事实输入。
- Final 返回 O/C/E/A/N 五档类别或 null、reasoning，以及引用有效 M 记录的 contextual 表现。

有输出 Schema 的请求开启 API JSON Object 模式，同时在提示词中给出实际 Schema。服务端 JSON 模式不等于业务正确性：时间顺序、窗口范围和来源仍由 Python 检查。原始响应及错误均保留，不解析旧别名或恢复截断 JSON 前缀。

请求不自动重试 HTTP 错误。业务格式修正沿用当前配置的有限次数，候选修正只处理失败项，不增加重试层级。默认输出上限 8192 tokens，不设置候选或证据条数上限。音频完整发送，视频发送已有 FPS 采样帧；传输限制导致失败时不会静默截断媒体。

## 日志与质量判定

每轮保存 config.json、source_hashes.json、去除监督字段的 manifest.jsonl、trajectories.json、rejected.json、api_calls.json、summary.json 和原始 generations。日志不包含认证头或媒体 Base64。

API usage、模型返回版本、耗时、finish_reason 与截断状态保留；金额未经核对时保持 null。本轮使用 Offline Scorer，Jev 请求为零，不进行训练。

流程检查包括粗扫优先、实际动作去重、候选时间合法性、Memory 来源一致性和 Final 引用。质量检查另看行为具体性、重复内容、文本来源支持及情境区分；未核实的声音、表情和人格解释不能因 JSON 合法而被视为正确。

所有诊断轨迹默认 teacher_accepted=false。只有来源与语义审核、Train 归属及任务表现通过后，才可用学生 tokenizer 编码模型生成的目标文本并导出 SFT；不会把 API Token ID、Python 工具结果或候选计划当作已观察事实。

官方接口参考：[模型说明](https://help.aliyun.com/en/model-studio/qwen3-8-omni-flash)、[Chat Completions](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions)、[结构化输出](https://help.aliyun.com/en/model-studio/qwen-structured-output)。本文描述当前项目实现；在线验证结果另见本轮审计记录，不把模拟测试视作真实模型验证。

## 2026-11-25 真实 API 诊断结果

使用同一当前框架和 qwen3.8-omni-flash，reasoning_effort=medium、输出参数 max_tokens=8192、最多 64 次主动观察。每个数据集仅生成一条轨迹，不是 best-of-N，也不是正式性能实验。输入包含完整访谈，未向 API 传入人格真值或旧模型答案。

| 数据集 | 主动观察 | API 调用 | 结束原因 | 参数修正 | Final |
| --- | ---: | ---: | --- | ---: | --- |
| IELTS | 46 | 56 | 候选池耗尽 | 0 | 已生成 |
| RecruitView，人物级多问答 | 64 | 80 | 64 轮预算耗尽，仍有 7 个候选 | 1 | 已生成 |

两条轨迹均通过：粗扫候选先于细化执行、实际动作不重复、Memory 对应实际观察来源、JSON 可解析、Final 引用编号合法。无输出截断或未修正候选错误。两条粗扫均有一个来源返回空候选；程序没有填充它们。RecruitView 的一次局部时间越界由原有有限修正流程处理。当前模型仍可能提出冗余目标，协议完成不等于决策高效。

### 质量审核结论

API 能描述具体问答、行为自述和可见性限制，内容比本地 3B 的人名式片段丰富。但这些轨迹尚不具备直接作为高质量教师数据的条件：

- 反复读取同一录制行为，不能据此获得多份独立的人格支持。IELTS 的 Final 存在这种归因问题。
- 模型输出了 F0、jitter/shimmer、毫秒级起声延迟、头部角度等结论；框架没有提供相应测量工具结果，不能将这些描述视为已核实测量。
- 某些音频记录把空 segments 数组当作“没有词语”的证明。音频观察没有运行 ASR；没有转写标注不能证明静音或无词语内容。这是需要进一步澄清的输入/解释边界。
- IELTS 的人格解释受语言能力、答题任务影响；RecruitView 的解释引入了未独立证实的背稿、投射等假说。具体行为自述与这些推断需要分开审核。
- IELTS 产生 222 条不同措辞的观察字符串，RecruitView 产生 344 条；这些不是经验证的独立人格证据数量，零完全相同字符串也不代表没有语义重复。

离线核对同一个 IELTS Dev 视频，按项目既有五档标签映射，五维命中 2/5，序数平均绝对误差 1.2。这只是一个样本，不能作为测试集准确率。标签在推理完成后读取，没有进入 API 输入。RecruitView 尚无核实的人物级标签，本轮没有将逐回答标签平均成人物真值，也没有报告其分类准确率。

### 使用边界与后续工作

本轮共 136 次真实 API 调用。实际 token usage 保存在本地审计记录，金额没有核对，保留 null；没有进行 Jev 调用、SFT 导出或正式训练。两条轨迹均保持 teacher_accepted=false。

下一步应先澄清当前媒体、未执行 ASR 和历史未审核笔记的区别；让 Final 根据不同的行为事实及情境判断，保留不确定性，避免将重复读取或未经测量的数值作为独立支持。还需要提高细化目标的信息价值，不能靠持续增加预算解决冗余。

这些问题不要求恢复复杂来源 ID 输出、Memory 语义执行器或增加新的 Agent。当前 Memory 仍按既有协议保存原文，事实与人格语义质量需要后续审核。

详细原始响应、访谈、标签和个人预测仅保留在本地 runs/api_latest_validation/，未上传到 GitHub。仓库只提供这份去除个人内容的汇总。
