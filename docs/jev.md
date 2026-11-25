# Jev 辅助评分

当前使用 `TraitScorer` 接口与官方 SDK 实现 `JevTraitScorer`，模型版本由 `jev_model` 配置，当前固定为 `jev-1.14.0`，SDK 固定为 `typesafe-sdk==0.7.5`。接口说明见[官方 Score 文档](https://docs.typesafe.ai/primitives/score)。

## 运行

使用专用 Conda 环境。通过环境变量配置 `TYPESAFE_API_KEY`，密钥不写入代码或配置；确认数据外发权限后才使用：

```bash
bash scripts/test_local_omni3b.sh --scorer jev --allow-external --output runs/jev_check
```

默认 `offline` 仅用于流程测试，均匀概率标记为非预测，不能作为真实 Jev 结果。

## 数据与输出

Stage 1 只发现候选，不写入 Memory。每次 Observe 保存一条完整的 M 记录，其中可以有多条行为发现。Jev 输入只取 observations 中实际记录的发现及来源模态，不包含候选计划、人格标签、人物 ID、路径、精确时间或原始媒体。邮箱和电话会被遮盖，但不能保证自由文本完全匿名。

空发现不发起真实请求；读取内容无变化时复用评分。结果保存五个 Trait 的 level、五档概率、confidence、版本、原始响应、缓存状态、usage 和费用信息。五档索引按 very_low / low / medium / high / very_high 映射；未知等级允许 null。概率详情保留在独立 score 和 scoring_history 中。

评分指令要求按不同的行为及情境使用支持，重复或互补读取不能增加独立行为支持次数。输入仍保留所有实际发现，不由 Python 根据文本相似度合并事件；该指令是否被真实 Jev 正确执行仍需 Dev 集验证。

未知实际费用保留 null，不将 token 用量当作真实账单。缓存和有限 SDK 请求重试由现有 Scorer 管理，鉴权失败不会静默切换成离线预测。

## 训练边界

Jev 是辅助评分器，不是真值。本轮教师 API 测试使用 Offline Scorer，不调用 Jev。

当前 Memory 是观察记录，不具备经过验证的逐 Trait / Context 原子事件归因。因此局部 Trait/Context 奖励尚不可用，--enable-trait-reward 会明确拒绝；不能把前后均匀先验的变化当作真实改善。Dev 预测与校准工具保留，用于后续研究验证，不代表局部奖励已经启用。

当前可训练的 Memory 信用是独立来源检查的忠实性反馈；反事实 Memory Utility 未实现。真实行为与整体人格标签方向不同，不自动构成错误证据。
