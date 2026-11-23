# Voronoi 密度关系验证（2026-11-23）

方法公式及运行入口见 [说明](../docs/VORONOI_DENSITY_GRAPH_TRANSFORMER_zh.md)。
在原 master 增加独立方法 `voronoi`；旧 `irgraph` 的含义和 checkpoint 参数保留。
未更改 backbone、正式 batch size、原始数据、sources 或历史实验结果。

## 自动测试

查询融合、新旧图编码、三个 adapter、DIST、设备选择及运行入口的联合回归：**225 项通过**。
随后补充近重复时刻和默认第一批入口，共增加 3 个用例；最终新方法测试文件 **34 项全部通过**。
合计覆盖 194 个既有用例和 34 个新用例，228 个不同用例通过。

验证内容：

- 内部 Voronoi 密度、单边对称延拓端点、同变量重复时刻、singleton 中性编码。
- 每个变量频率不同的规则采样，其结构编码与密度关系项精确为零，含 float32/float64 舍入情形。
- 密度等于基准不意味着左右等间隔；(1,3)、mu=2 被正确编码为零。
- 平移、正尺度变换、反转、token 排列、batch 独立性。
- 对数高斯密度关系非正、对称，且规则端与非规则端仍有非零关系。
- 近重复正间隔产生有限、不被旧 [-8,8] 裁剪的密度编码；无效 NaN padding 被忽略。
- gamma 初始化为 1，经 softplus 保持非负方向；梯度正常传播。
- 一层/两层图的接收端分块、训练重计算与非分块预测及梯度一致。
- 新模型不存在旧 gap MLP、左右密度 Q/K 投影或 X-to-Y 密度 K 投影参数。
- 相同权重下，规则输入的 `voronoi` 与其 `voronoi_no_density` 消融输出完全相同。
- 三个模型保留原历史编码器和预测头，未来 Y 真值不影响预测；已提供未来 X 能影响更早查询。
- 全部 X 缺失时，查询融合为恒等映射。
- 三批脚本各跑五个数据集，默认只跑第一批 t-PatchGNN；固定 seed 3407，显式 cuda:1，不强设 CUDA_VISIBLE_DEVICES。
- 原配置 batch size 与显式 BATCH_SIZE 覆盖、含空格路径、拒绝覆盖既有结果目录。

另从修改前 HEAD 动态加载旧 `irgraph` 参考实现，严格加载相同 state_dict，验证两层不规则输入输出 **逐元素完全相同**。
新方法可复用旧时间/变量 attention，但不改变旧方法的预测行为。

## 真实数据 CPU 短程训练

- `results/voronoi_smoke_20261123`：三个模型 × Activity、MIMIC、PhysioNet、USHCN、DIST，**15/15 成功**。
  诊断小模型，每 split 一个样本/窗口，一个 epoch，batch=1，seed=3407。
- `results/voronoi_default_shape_20261123`：三个模型 × MIMIC、DIST，**6/6 成功**。
  保留正式模型宽度/层数，只在诊断命令中限制每 split 一个样本/窗口、一个 epoch、batch=1。

合计 **21 次真实数据训练/验证/测试与导出成功**。输出为新的忽略目录，未将诊断结果提交为正式实验结果。
本地 PyTorch 2.14.1，CPU 执行；短程指标仅验证可运行性，不表示新方案优于旧方法。
服务器 CUDA、完整数据精度和显存峰值仍需正式实验验证。

## 静态检查

修改及新增 Python 文件的关键语法/名称检查、shell 语法检查、`git diff --check` 通过。
没有改动用户原有四个未跟踪文件：GRAPH_TRANSFORMER_GRID_zh.md、processed_data_audit.json、
grid_search_exen_graph_transformer.py、test_graph_transformer_grid.py。
