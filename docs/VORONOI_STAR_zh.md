# Voronoi-Graph Former（voronoi_star）

新增独立实验方法 `voronoi_star` 与同结构对照 `voronoi_star_no_density`。
已有 `voronoi`、`voronoi_h`、`voronoi_h_4d` 的模型定义和检查点语义不变。
本方法尚无正式预测精度结论。

## 运行

在仓库根目录、已安装依赖的环境中：

```bash
# 单独跑 Hi-Patch–MIMIC，两种密度设置，batch size=16。
COMPARE=1 BATCH_SIZE=16 bash scripts/run_voronoi_star_query_fusion.sh hipatch mimic --device cuda:1

# t-PatchGNN，五数据集，成对对照。
COMPARE=1 bash scripts/run_voronoi_star_query_fusion.sh tpatchgnn all --device cuda:1

# 只运行一个版本。
BATCH_SIZE=16 bash scripts/run_query_fusion.sh hipatch mimic voronoi_star --device cuda:1
```

新方法默认两层星形传播，两种密度版本层数一致。沿用 seed=3407、最多 30 轮、
各模型配置中的早停设置和批大小；`BATCH_SIZE`
显式覆盖训练与评估批大小。`RUN_DIR` 可指定新目录，已有目录拒绝覆盖。
`DRY_RUN=1` 仅生成计划。`all` backbone 表示原三模型，`all_six` 包含新增的三种
不规则时序 backbone。数据集可选 activity/mimic/physionet/ushcn/dist。
输出根目录为 `results/query_fusion/voronoi_star_<backbone>_<timestamp>_<pid>/`。

## 算法与符号

编码器输入仍仅为任务允许的外生观测 X；内生历史经过原 backbone，原预测头保留。
节点初始编码是 `value_linear + variable_embedding + time_features`，在 d 维内容
通道上做 LayerNorm。密度不拼入节点，不进入原 4d 联合归一化。

对真实接收观测 i，以其时间 t_i 为中心，在每个变量 c 上寻找严格小于和大于
t_i 的最近时间组。连接这些时间组中的所有观测，形成有向消息边 j→i。
还连接同刻观测（包含自身）。变量 ID 只表示类别，不参与距离计算。

- 双侧可用：`L_i|c = (delta_left + delta_right)/2`，`rho_i|c = 1/L_i|c`。
- 单侧可用：沿用镜像延拓，`L_i|c = positive_gap`。
- 没有正间隔：该变量的密度未知，保留同步内容边，密度分支对该边为零。
- 重复时间：几何使用唯一时间；被选时间组的所有值都参与聚合，不按 token
  顺序任取一个。同步边不造成 L=0。大批重复观测会增加实际图度数。
- 星形邻居包含自身变量；单变量时 L 退化为标准一维 Voronoi 长度，包括镜像边界。

各接收节点共享同一套几何定义，但多层传播时消息携带的是更新后的隐藏特征。
只存每节点的邻居索引及 `[B,S,V]` 密度场，不建立 `[B,S,S]` 全连接邻接矩阵。

## 训练集参考密度

参考密度按接收变量 a 和来源变量 c 条件化：

`bar_rho[a,c] = mean_{training X receivers i: c_i=a, rho_i|c known} rho_i|c`。

同一训练观测若出现在多个任务窗口中，按其在任务窗口中的出现次数参与统计。
参考的中心分布与该接收变量的观测时刻一致，并非对 c 的原生观测密度直接取均值。
计算在 float64 对数域累积，避免先取倒数造成溢出。

`TPatchForecastTrainer.train` 在第一轮前，以独立、不洗牌且不丢末批的 loader
扫描训练数据一次。只提取合法 X，排除预测标签、Y 历史和无效值；不调用模型
forward、不访问验证/测试集，不消耗训练 loader 的随机排列。恢复已有已标定
检查点时跳过扫描。

`log_reference`、`reference_count`、`reference_ready` 为模型 buffer，随模型和
检查点保存；forward/eval 不更新它们。训练中未出现的 (a,c) 对视为未知，不用
当前验证/测试窗口补估，也不把未知当成 r=1。手动使用编码器时须先调用
`fit_reference(training_observations_iterable)`，或载入已标定 state_dict。

## 密度场与学习分支

仅在几何及参考均可用的变量集合 C_i 上计算：

```text
r_i|c = rho_i|c / bar_rho[c_i,c]
s_i|c = log(r_i|c)
p_i|c = r_i|c / sum_v r_i|v
a_i|c = log(|C_i| * p_i|c)
S_i   = log(|C_i| / sum_v 1/r_i|v)
z_i|c = [s_i|c, a_i|c, S_i]

eta_i|c = MLP_context([a_i|c, S_i, Emb_variable(c)])
u_i     = mean_c eta_i|c
```

份额是“相对参考密度”的份额，不是观测次数比例。S 是相对密度调和平均的
对数，也等于平均标准化 Voronoi 长度的负对数。S 同时出现在 z 和 eta 路径，
保持讨论中的设计；这种重复条件化是否必要尚待消融。

对每个注意力头 h、边 j→i：

```text
q_i, k_j, v_j = linear projections of content-only LN(h)
e_base_ij = q_i · k_j / sqrt(d_head)
            + original time-embedding and variable-relation biases
b_ij = MLP_edge([q_i, k_j, z_i|c_j, u_i, Emb_edge_type])  # d_head vector
e_ij = e_base_ij * ones(d_head) + lambda_head * b_ij
alpha_ij[channel] = softmax_{j in star(i)} e_ij[channel]
message_i = sum_j alpha_ij * v_j
h_i_new = residual + output_projection(message_i), followed by residual FFN
```

边类型为同/跨变量 × 前/后/同步六类。lambda 是可学习向量，初始化为零；边网络
正常随机初始化，首步可学习门控，随后学习边网络和上下文。无可用密度的边
直接屏蔽 b，不把零特征当作已知正常密度。无可用密度变量时 u=0。

`voronoi_star_no_density` 保留相同图、模块与参数总数，显式绕过密度分支，
不会依赖 lambda 是否恰为零。门控为零且权重相同时，两种版本输出完全一致。
这个保证只适用于相同的星形图：不意味着等价于旧的全连接 no_density 模型。
训练后，即便所有相对密度为 1，边网络仍可能利用 q/k 和变量身份改变注意力；
本实现不宣称“均匀采样时始终严格退化”。

## 验证与解释边界

`tests/test_voronoi_star.py` 覆盖解析长度、严格邻居、单变量退化、同步与重复
时间、份额/全局水平、训练参考保存、无密度对照、独立标量注意力参照、分块
反向梯度、标签隔离，以及原三 backbone 的接入和启动计划。

几何公式有 Voronoi 依据；参考标定、调和摘要、DeepSets 和边网络是明确的
架构选择。功能测试不证明预测收益或文献创新性，正式指标需要重新运行实验。
