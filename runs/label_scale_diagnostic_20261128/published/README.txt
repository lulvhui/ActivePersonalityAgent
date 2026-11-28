结论：存在方向性的排序信息，O 最一致；C 仅在混合全部运行时呈正相关，E 较弱，A 很弱。仅 6 位 Train 被试，尚不能据此确认可泛化的排序能力。
N 的问题不能归结为可用排序上的尺度偏移：20 条预测中 18 条 low、1 条 very_low、1 条 medium；标签全部 medium 及以上。N 的 Spearman=0，每人一条时全部 low、相关不可定义。存在明显尺度/位置偏移，但当前离散输出没有展示 N 的个体排序信息；无法仅凭这些输出区分感知、推理、阈值/量化分别贡献多少。
秩校准使全部轨迹 accuracy 0.150→0.190、ordinal MAE 1.280→1.120；多数类基线为 0.350 / 1.260。MAE 改善主要来自把 N 整体上移，并不能作为恢复 N 排序能力的证据。

标签尺度诊断（2026-11-28）
============================

范围与处理
- 四组指定实验中，非 offline 且具备五特质 Final 结构的 Train 轨迹共 20 条、6 人：原始 pilot 4、bounded 5、grounded 5、seed 6。保留 partial（原始 183/193、grounded 183、seed 167 r1）。
- 原始 pilot 的 106 和未恢复的 176 归档中 trajectories.json=[]，没有 Final，未计入；176_resume 是单独保留的真实续跑，不能当作完全独立的重复实验。
- 主仓库、本分支中相同文件的归档/本地副本按完整轨迹内容 SHA256 去重；同人不同运行全部保留。complete_candidate_packet 是现有轨迹副本，不作为新运行。
- 另找到 6 条 Dev 轨迹：IELTS:123 一条、RecruitView:1703 三条、RecruitView:person:327 两条。前四条单独用 dev.jsonl 评估；后两条没有可对应的人级标签，只清点，绝不从 answer 标签生成 person 目标。
- null 按 amp.evaluate 逐特质弃权：conditional accuracy/MAE/相关排除 null，accuracy_all 将弃权计错。Train 100 个预测均非 null；RecruitView Dev 的覆盖/弃权详见 report.json。
- 每人一条的规则不看标签：按实验名选择最后一组，再取该组字典序第一条（seed 取 r1，包括 partial）；其余 138/176/193 取 grounded。不是 Best-of，也没有排除困难个体。
- 标签为 train 内各数据集、各特质分位数，经 bisect_right 划五档；原始标签 ties 会使档位不严格等频。等级编码 0=very_low、1=low、2=medium、3=high、4=very_high。

相关：1000 次置换，seed=20261128，p=(超越次数+1)/1001
下列 naive p 将轨迹当作独立；cluster p 把同一人的五特质 raw 标签向量一起在 6 人之间置换，保留重复运行和特质间依赖。
p+ 为正向单侧，p2 为双侧。naive 小 p 不应解释为 20 个独立样本的证据；未作多重比较校正。
特质  全轨迹rho  naive_p+ naive_p2 cluster_p+ cluster_p2 每人一条rho 每人p+ 每人p2
O  0.621  0.004  0.005  0.053  0.100  0.636  0.145  0.276
C  0.447  0.024  0.048  0.044  0.115  NA  NA  NA
E  0.340  0.078  0.149  0.191  0.368  0.422  0.364  0.710
A  0.138  0.285  0.543  0.345  0.697  0.220  0.395  0.807
N  0.000  0.692  1.000  1.000  1.000  NA  NA  NA
五特质合并 train_all: n=100, rho=0.437, p+=0.001, p2=0.001
五特质合并 train_one_per_person: n=30, rho=0.528, p+=0.002, p2=0.003
合并五特质会混入特质间 raw 尺度/位置和预测分布差异，不能证明每一特质有个体排序能力；每人一条的 30 对也不是 30 个独立人。逐特质结果优先。
O 在四个实验组内均为正相关（0.236/0.725/0.889/0.645），是最一致的方向性信号；cluster 双侧 p≈0.100。C 混合运行的正相关在每人一条时消失（全 medium）；seed 中 A 甚至为负相关，混合不同提示/版本不能当作单一固定 teacher 的验证。

分布与秩校准
秩校准公式：p_new=min(4,floor(5*(平均零基秩+0.5)/n))，每特质在该评估子集内拟合，不使用标签值。ties 必须保持同档，所以不能强行做到等频；常数 low 会整体变成 medium。不是可部署校准器，也不是 held-out 准确率。
原始/校准指标均在相同非弃权对上算；多数类为全 Train 各数据集各特质众数（并列取较低档），不在诊断小样本内拟合。
特质  预测5档计数  标签5档计数  原始acc/MAE  校准acc/MAE  多数类acc/MAE
O  [0, 10, 9, 1, 0]  [0, 0, 13, 3, 4]  0.150/1.100  0.100/1.000  0.200/1.450
C  [0, 2, 16, 2, 0]  [0, 3, 7, 7, 3]  0.400/0.700  0.450/0.700  0.150/1.500
E  [1, 11, 8, 0, 0]  [2, 0, 0, 18, 0]  0.000/1.550  0.000/1.550  0.900/0.300
A  [0, 1, 14, 5, 0]  [6, 0, 4, 10, 0]  0.200/1.100  0.150/1.300  0.500/1.100
N  [1, 18, 1, 0, 0]  [0, 0, 5, 11, 4]  0.000/1.950  0.250/1.050  0.000/1.950
train_all 合并 accuracy/ordinal MAE: original=0.150/1.280; rank_calibrated=0.190/1.120; majority_same_covered_pairs=0.350/1.260
train_one_per_person 合并 accuracy/ordinal MAE: original=0.133/1.233; rank_calibrated=0.267/1.100; majority_same_covered_pairs=0.333/1.300
N 校准后 accuracy=0.250，MAE=1.050，原始为 0.000/1.950。每人一条时 N 常数 low→常数 medium，MAE 1.833→0.833。这个收益无需任何个体排序，不能据此认定 N 只是尺度问题。

同人两次运行的一致率
相同实验内 seed 的 r1/r2 共 3 对：O 2/3、C 3/3、E 2/3、A 3/3、N 3/3；合并 13/15=86.67%。
所有版本同人运行两两比较共 26 对：O 19/26、C 18/26、E 15/26、A 16/26、N 19/26；合并 87/130=66.92%。这些对彼此相关，跨版本差异混入提示和实现变化，不能叫作同配置独立重复性。N 的高一致率主要反映输出塌缩。

轨迹来源与 tested commit
完整路径、归档成员、同文件副本别名、每条 sample_id/trait/预测等级/标签等级都在 report.json inventory。
已记录的 acquisition commit 与推断的代码快照严格分列；旧实验未保存运行 HEAD 时不伪造 tested commit。
- teacher_pilot5_20261126/IELTS_138.tar#0: IELTS:138; complete; train; tested commit: 未记录；源码哈希唯一匹配 462e3b8658db15769534020cd146b1451729e252
- teacher_pilot5_20261126/IELTS_176_resume.tar#0: IELTS:176; complete; train; tested commit: 未记录且无法唯一还原
- teacher_pilot5_20261126/IELTS_183.tar#0: IELTS:183; partial; train; tested commit: 未记录；源码哈希唯一匹配 462e3b8658db15769534020cd146b1451729e252
- teacher_pilot5_20261126/IELTS_193.tar#0: IELTS:193; partial; train; tested commit: 未记录；源码哈希唯一匹配 462e3b8658db15769534020cd146b1451729e252
- api_flow_validation_20261124/recruitview327#0: RecruitView:person:327; partial; unscorable_person_target; tested commit: 未记录且无法唯一还原
- api_latest_validation/IELTS_123#0: IELTS:123; complete; supplementary_dev; tested commit: 未记录；源码哈希唯一匹配 1d4897a711bf9be2d864a90812e8552d4c2aa292
- api_latest_validation/RecruitView_person327#0: RecruitView:person:327; complete; unscorable_person_target; tested commit: 未记录；源码哈希唯一匹配 1d4897a711bf9be2d864a90812e8552d4c2aa292
- memory_cleanup_validation/api_recruit1703#0: RecruitView:1703; partial; supplementary_dev; tested commit: 未记录且无法唯一还原
- qwen38_api_teacher/qwen38_api_teacher#0: RecruitView:1703; partial; supplementary_dev; tested commit: 未记录且无法唯一还原
- qwen38_api_teacher_retest/qwen38_api_teacher_retest#0: RecruitView:1703; complete; supplementary_dev; tested commit: 未记录且无法唯一还原
- teacher_pilot5_bounded_20261127/IELTS_106.tar#0: IELTS:106; complete; train; tested commit: abe7c9639603ffa15d48ad966b9afe2f99ec3bd5
- teacher_pilot5_bounded_20261127/IELTS_138.tar#0: IELTS:138; complete; train; tested commit: abe7c9639603ffa15d48ad966b9afe2f99ec3bd5
- teacher_pilot5_bounded_20261127/IELTS_176.tar#0: IELTS:176; complete; train; tested commit: abe7c9639603ffa15d48ad966b9afe2f99ec3bd5
- teacher_pilot5_bounded_20261127/IELTS_183.tar#0: IELTS:183; complete; train; tested commit: abe7c9639603ffa15d48ad966b9afe2f99ec3bd5
- teacher_pilot5_bounded_20261127/IELTS_193.tar#0: IELTS:193; complete; train; tested commit: abe7c9639603ffa15d48ad966b9afe2f99ec3bd5
- teacher_pilot5_grounded_20261127/IELTS_106.tar#0: IELTS:106; complete; train; tested commit: 3b8543dcf51505244761f1d38aa702c80f1d88f5
- teacher_pilot5_grounded_20261127/IELTS_138.tar#0: IELTS:138; complete; train; tested commit: 3b8543dcf51505244761f1d38aa702c80f1d88f5
- teacher_pilot5_grounded_20261127/IELTS_176.tar#0: IELTS:176; complete; train; tested commit: 3b8543dcf51505244761f1d38aa702c80f1d88f5
- teacher_pilot5_grounded_20261127/IELTS_183.tar#0: IELTS:183; partial; train; tested commit: 3b8543dcf51505244761f1d38aa702c80f1d88f5
- teacher_pilot5_grounded_20261127/IELTS_193.tar#0: IELTS:193; complete; train; tested commit: 3b8543dcf51505244761f1d38aa702c80f1d88f5
- teacher_seed_c56ec92_20261127/IELTS_106_r1.tar#0: IELTS:106; complete; train; tested commit: c56ec929f5212319e06663d7dee31bf977d48c01
- teacher_seed_c56ec92_20261127/IELTS_106_r2.tar#0: IELTS:106; complete; train; tested commit: c56ec929f5212319e06663d7dee31bf977d48c01
- teacher_seed_c56ec92_20261127/IELTS_167_r1.tar#0: IELTS:167; partial; train; tested commit: c56ec929f5212319e06663d7dee31bf977d48c01
- teacher_seed_c56ec92_20261127/IELTS_167_r2.tar#0: IELTS:167; complete; train; tested commit: c56ec929f5212319e06663d7dee31bf977d48c01
- teacher_seed_c56ec92_20261127/IELTS_183_r1.tar#0: IELTS:183; complete; train; tested commit: c56ec929f5212319e06663d7dee31bf977d48c01
- teacher_seed_c56ec92_20261127/IELTS_183_r2.tar#0: IELTS:183; complete; train; tested commit: c56ec929f5212319e06663d7dee31bf977d48c01

执行与复现
- 原 worktree 的 Git 元数据在只读主仓库路径，git pull --ff-only 因 FETCH_HEAD 不可写失败。改用 /tmp/label-scale-checkout-20261128 独立检出 feature/bounded-refinement 并 git pull --ff-only，核验 HEAD=4cdc434aadee80cf62732f2c9238ac8be7b79f26。评估和单元测试在此版本执行；原 worktree 的 Git HEAD 因权限限制未更新。
- 未修改 amp/，没有 API/模型调用，也没有读取 .env 或其他凭据。标签仅在本地评估脚本与 evaluate 中使用，未进入 Agent/Memory/Jev。
- 归档只向 TemporaryDirectory 解出 trajectories.json、summary.json、source_hashes.json；合并完整轨迹和 evaluate 临时结果也在其中，脚本退出即删除。published 仅保留本 README、聚合 report 和分析脚本，不包含完整轨迹、raw 标签逐条数值、媒体或凭据。
- 标准 evaluate 已分别运行于全部 Train、每人一条 Train 和有标签 Dev；原样结果嵌入 report.json 的 *_amp_evaluate 字段。主输入为 /mnt/r6_hdd4t/lhlu/ActivePersonalityAgent/runs/data/train.jsonl、label_bins.json，Dev 补充用同目录 dev.jsonl；文件 SHA256 已记录。
- python -m unittest discover -s tests: Ran 195 tests，OK (skipped=9)，测试版 4cdc434；测试日志哈希在 report.json validation。

复现示例（从 4cdc434 检出根目录执行，PATH 中需有 python；仅运行本地统计）：
  PYTHONPATH="$PWD" python <published>/scripts/diagnose.py \
    --branch-root /mnt/r6_hdd4t/lhlu/ActivePersonalityAgent_branch_validation/bounded_refinement_ddebacd \
    --main-root /mnt/r6_hdd4t/lhlu/ActivePersonalityAgent \
    --output <new-output-dir>
可选 --unit-test-log <已运行的unittest日志> 会附入测试汇总。

局限：极小的非随机 Train 选择集；重复人物、多套提示/版本、人工续跑和离散等级 ties；置换检验是探索性证据，非确认性推断；无原始媒体语义审计，无法因果定位感知/推理阶段。没有批准任何轨迹用于训练。
