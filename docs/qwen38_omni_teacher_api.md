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
