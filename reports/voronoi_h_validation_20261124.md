# Voronoi 节点编码验证（2026-11-24）

本次增加 `voronoi_h` 和匹配消融 `voronoi_h_no_density`，见 [公式与运行说明](../docs/VORONOI_NODE_ENCODING_zh.md)。
直接在原 master 工作，旧 `voronoi`、`irgraph`、backbone、数据及历史结果均保留。

## 实现

密度计算与已推送的 attention 版本完全相同：内部 $\rho=2/(\delta^L+\delta^R)$，
首尾使用对称延拓下的 $\rho=1/\delta$，特征 $s=\log(\mu\rho)$。
新节点输入为 $[\operatorname{Linear}(x)\Vert e\Vert\phi(t)\Vert s]$，
经过 Linear(3d+1,d) → SiLU → Linear(d,d) 得到 h。
取消独立密度 logit，不实例化 gamma 或左右密度 Q/K 投影。
密度通过 h 影响 Q、K、V，X→Y 不添加额外密度项。

`voronoi_h_no_density` 与主方法具有完全相同的参数结构，仅将 s 输入置零。
因此可区分密度通道作用与非线性融合容量变化。
规则采样的 s 仍为零，新方法与相同权重的匹配消融严格一致，但不声称等同旧加和编码器。

## 自动测试

以下联合回归 **257 项全部通过**，其中 Voronoi 相关文件覆盖 63 项：

```text
tests/test_voronoi_graph_transformer.py
tests/test_irregular_graph_transformer.py
tests/test_query_fusion.py
tests/test_strgcn_query_fusion.py
tests/test_mtan_query_fusion.py
tests/test_dist_query_fusion.py
tests/test_query_fusion_runner.py
tests/test_device_selection.py
```

新增/扩展覆盖：

- 节点融合输入维度、密度通道权重获得非零梯度，图 Q 和 backbone 梯度正常。
- 模型中不存在独立密度偏置 gamma、旧 gap MLP 或左右密度 Q/K/XY 投影。
- 规则密度为零时匹配消融预测完全相同；不规则密度通道影响输出。
- 一层/两层图、分块 checkpoint 与非分块输出及梯度一致。
- token 排列、无效 NaN padding、无 X 时恒等映射，未来 X 可以影响更早查询。
- 三模型 × 四公开数据集原预测头接口与未来 Y 标签隔离。
- 三模型 × DIST 的 h 主版本和匹配消融均可前向/反向。
- 三批脚本默认 t-PatchGNN，仅跑一个模型的五个数据集；COMPARE=1 对应十次匹配消融。
- 保留配置 batch size、固定 seed 3407、显式 cuda:1、含空格路径与拒绝覆盖旧结果。

从修改前 HEAD 动态载入旧 `irgraph` 和旧 `voronoi`，严格加载其 state_dict，
两层不规则输入的预测均与本次代码 **逐元素完全一致**。
脚本语法、关键 Python 名称/语法检查和 `git diff --check` 通过。

## 真实数据 CPU 训练

`results/voronoi_h_smoke_20261124`：三个模型 × Activity、MIMIC、PhysioNet、USHCN、DIST，
诊断小模型、每 split 一个样本/窗口、一个 epoch、batch=1，**15/15 成功**。
逐项检查 `metrics.json` 的 `protocol.fusion`，确认 density_location=node、
density_node_encoding=true、density_attention_encoding=false、density_bias=none。

`results/voronoi_h_default_shape_20261124`：三个模型 × MIMIC、DIST，保留正式模型宽度/层数，
仅在诊断命令限制每 split 一个样本/窗口、一个 epoch、batch=1，**6/6 成功**。
合计 21 次真实数据训练、验证、测试与导出通过。

以上均为本地 PyTorch 2.14.1 CPU 检查，不用短程指标推断新旧方法优劣。
正式 GPU 实验仍需用户在服务器上运行，默认单 seed 3407，不改变正式 batch size。
所有本次诊断输出均写入新的忽略目录，不覆盖或提交历史实验输出。
用户已有四个未跟踪文件未改动、未提交。
