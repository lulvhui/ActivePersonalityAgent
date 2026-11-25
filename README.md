# ActivePersonalityAgent

当前流程：Global Coarse Perception → 程序调度 Select → Observe → 保存观察记录 → Contextual Personality Reasoning。

## 环境与启动

运行环境为 `~/miniconda3/envs/active-personality`，不使用 venv。安装器只在专用环境安装项目依赖，base 不安装项目包。

```bash
bash scripts/setup_local_omni3b.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate active-personality
bash scripts/test_local_omni3b.sh --check-only
```

本地 Qwen2.5-Omni 使用 BF16，关闭语音生成，Transformers 为 4.57.1。默认不调用外部 API。视频按 FPS 采样，音频完整解码，处理器音频截断关闭。

在其他机器运行前，修改所用配置中的 `data_root` 和 `model`，分别指向既有数据目录与本地模型权重。GitHub 源码包中的配置使用相对路径占位，不包含数据、模型或凭据。API 入口及凭据说明见 [教师 API 文档](docs/qwen38_omni_teacher_api.md)，Jev 辅助评分见 [Jev 文档](docs/jev.md)。

## 当前输出协议

粗扫描覆盖完整媒体，默认按 20 秒传输窗口、1 FPS 视频及连续音频进行。传输窗口不自动变成待办任务；候选只来自模型，允许空数组。

```json
{"summary":"当前来源的简短概要","candidates":[{"hint":"实际线索","goal":"要检查的行为细节"}]}
```

Python 把根候选绑定到当前来源的真实窗口和模态。模型不生成来源 ID、文本片段编号或粗扫时间。相同物理读取保留一个任务，多个 hint/goal 全部保存在 leads 中，不限制线索数量。

Select 默认由程序调度待办，先读取全部可执行的粗扫候选，再按提出顺序处理细化任务，不调用模型，也没有 STOP 输出。细化候选在粗扫候选读取时可以入池，但不会抢占尚未读取的粗扫候选。程序根据候选池和预算结束观察，不设置最低轮数。

Observe 记录行为和情境，并可提出多个细化请求：

```json
{"observations":["受访者描述的行动及其情境"],"candidates":[{"local_time":[2,8],"modality":"audio","hint":"已观察线索","goal":"需要澄清的细节"}]}
```

`local_time` 是父片段从零播放时的秒数，不是全局时间或百分比。默认允许前后各 5 秒上下文延伸，并校验真实媒体范围、视频窗口长度和已执行任务。文本、音频、视觉是三个模态；`audio_video` 表示联合观察。

Memory 每次保存完整模型原文和 Python 生成的 `M1/M2…` 来源记录。一条记录可以包含多个发现，不等于一条独立人格证据。读取时只取 observations，不把 candidates 中的计划当作已观察事实。当前没有原子事件归并、人格方向筛选或 Memory Tool。

Observe 读取时间上相关的已保存发现作为上下文，并用候选目标指出尚待澄清的信息。候选参数修正只处理失败项，提供合法局部时间范围、视频窗口上限和现有请求状态；已经入池的合法候选不要求重新生成。修正仍使用原有有限次数，不增加重试层级。

Observe 输入统一使用当前片段从零播放的秒数。历史请求只展示当前上下文内的 `covered_local_time` 覆盖范围；历史记录的绝对时间不进入该生成任务。未验证 ASR 只提供文本及状态，已验证文本位置才提供局部时间。完整来源时间仍保存在 Python 的观察记录中。修正输入将每个失败候选与对应错误放在一起，避免子集索引与原列表索引混淆。

Final 输出 O/C/E/A/N 的五档等级或 null、共享 reasoning，以及带 M 记录引用的 contextual 表现。不同情境的行为差异不自动代表稳定人格发生变化。

输出格式在生成前告知模型，本地解码使用同一 Schema。原始输出始终保留；不再解析 Markdown、Python 字典、旧字段别名或截断 JSON 前缀。无效输出明确记录为 partial/unavailable，不补造结果。默认 Token 安全上限仍为 4096，没有新增摘要字符限制，也没有 Candidate/Evidence 条数上限。

## 数据与来源

```bash
python -m amp.cli prepare --output runs/data
python -m amp.cli infer --manifest runs/data/dev.jsonl --dataset IELTS --limit 1 --output runs/infer_ielts
```

使用既有按人划分的 Train/Dev/Test。RecruitView 推理将所选人物的多个原始问答视频拼接，并保留问题及时间映射。其人物级输入标记为 `person_unresolved`，不把逐回答标签自动平均成人物真值。正式人物级训练与评价必须先解决对应标签。

ASR 时间仍是未验证的检索提示，不作为精确媒体对齐真值。M 记录引用实际 Observe 的窗口和模态；来源合法不等于内容或人格解释已通过语义审查。

## 教师轨迹、SFT 与 TECA

保留 Teacher Trajectory SFT → GRPO/TECA 路线，当前统一使用观察记录 Memory。旧 Cue/原子证据/Memory Tool 轨迹不能直接导入，需按当前协议重新生成。

现有独立 Judge 按每个 M 记录检查实际来源中的发现，并评价 Final。检查结果与记录、来源指纹绑定；缺失或过期检查不能使教师轨迹通过筛选或获得正向 Memory 质量信用。真实标签仅用于离线筛选和训练奖励，不进入 Agent、Memory 或 Jev 输入。

教师筛选保留完整 Train 轨迹、标签任务表现及独立来源检查门槛。SFT 只监督模型实际生成的粗扫、Observe/候选计划和 Final Token；调度器及 Python 结果不是目标 Token。默认队列流程不伪造 Action 段。

```bash
python -m amp.cli export-sft --trajectories runs/teacher/trajectories.json --output runs/sft.jsonl --tokenizer /path/to/student
python -m amp.cli sft --data runs/sft.jsonl --output runs/sft
```

TECA 的 clipped GRPO 和冻结 SFT reference 保持不变。当前 Memory 段使用轨迹优势加来源忠实性信用：支持为正，不支持/错误来源为负，未知、空观察、重复计入旧更新为零。该信用不判断人格标签方向，不估算反事实效用。

当前记录没有可靠的逐 Trait/逐 Context 事件归因，因此局部 Trait/Context 增益不伪造数值，`--enable-trait-reward` 会明确拒绝。当前支持轨迹监督与 Memory 忠实性；这不等于完整事件级 TECA 已验证。`U_memory` 仍未估计。

`baselines/omniagent_native/` 是既有对照实验的适配代码，依赖另行安装的 [OmniAgent](https://github.com/HarryHsing/OmniAgent) 和独立推理环境；部分历史启动脚本仍需按本机目录调整，不是当前主流程的启动入口。仓库不打包其权重、外部代码库或访谈数据，也不替代第三方项目的许可要求。

Jev 的可替换接口、概率、缓存、费用和原始响应保留。仅显式授权外部访问后才能调用；它接收去标识化的已观察描述，不接收候选计划、人格真值或原始媒体。空描述不调用 API。正式使用概率反馈前仍需 Dev 能力及校准验证。

## 验证与结果

```bash
python -m unittest discover -s tests
bash runs/record_protocol_cleanup/run_local.sh
```

最后一次时间坐标清理后的 3B 完整复测在 `runs/record_protocol_cleanup/coordinate_full/3b/`，包含两条从粗扫开始的新轨迹；`validation.json` 记录源码一致性、参数修正及重复提议，不能与四个失败位置的 `coordinate_recheck/` 定点测试混淆。两条均在 27 次观察后自然耗尽候选池；IELTS 有一次零长度候选，经有限修正后撤回。

`flow_fix/7b/` 是同版时间输入的 7B 完整测试；`flow_fix/3b/` 是最后一次坐标清理之前的 3B 测试。`native/` 和 `budget64/` 为更早对照。最新本地完整测试预算为 64 轮，生产默认仍为 16，没有强制最低轮数。总报告为 `runs/record_protocol_cleanup/report.json`。单元、CPU 小策略参数更新、真实多模态推理和真实 Qwen 训练验证分别记录，不能相互替代。

清理前源码和淘汰协议测试保存在 `runs/record_protocol_cleanup/before/`；淘汰清单在 `retired_tests.json`。历史报告保留原始含义，不再作为当前接口说明。候选的信息价值、细微音视频描述及人格解释仍需语义复核，JSON 正确不代表人格证据可靠。
