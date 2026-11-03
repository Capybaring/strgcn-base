# DIST：CSR 稀疏关系计算

DIST 的模型配置已启用：

```yaml
model_params:
  relation_backend: sparse_csr
  receiver_chunk_size: 128
```

其他数据集及未指定 backend 的调用仍使用原 `dense` 实现；也可在各自的模型配置中选择 `sparse_csr`。
本次不重新训练已经完成的其他数据集。参数名称和形状未改变，旧权重可加载到两个 backend。

## 运行

拉取代码后，使用新输出目录：

```bash
conda activate strgcn-exen
python -u main.py \
  --config_path config/ExEn/processed/DIST.yaml \
  --model_save_dir_path model_states/exen_DIST_sparse \
  --result_save_dir_path results/exen_DIST_sparse
```

`train_config.device` 仍默认 CPU。有可用 CUDA 的机器请在配置中设为 `cuda:0`；仅安装 GPU 版 PyTorch 不会自动将此配置切换到 GPU。
实现使用原生 PyTorch CSR、sampled_addmm、segment_reduce 和 sparse.mm，无需额外图计算库。
已测试 PyTorch 2.15.0 的 CPU；CUDA 尚未实测。CSR 原生算子可能打印 beta 提示，它本身不是运行失败。
MPS 不支持此 backend，可使用 dense。

## 存储与消息传播

图的三类合法边全部保留：外生同变量时间边、内生同变量时间边、外生→内生边。
时间条件为 sender_time <= receiver_time；查询节点不发送消息；填充节点不参与。
没有邻居采样、Top-K、时滞截断、观测删减或超节点。

实现先按变量/关系划分邻居组，将已知源节点按时间排序，并用 searchsorted 得到每个接收节点的合法源前缀长度。
“源索引列表 + 接收索引列表 + 每行邻居数量”紧凑保存图拓扑，各层复用。
每个接收块只展开合法边的 CSR 列索引、logits 和权重，块间不保存完整邻接矩阵。
时间/变量内积使用 sampled_addmm 仅计算合法项，softmax 按 CSR 行归一化，消息使用 sparse.mm 聚合。
原来的源节点低秩变换提前计算一次，各接收块复用；训练仍使用块内重算控制中间激活。

它没有生成完整 L×L 布尔掩码、logits 或权重，也没有一次性将几千万条边全部展开成全局 COO。
空邻域保留原实现的自身更新/残差行为，跨变量阶段无入边的节点保留上阶段表示。
原编码、参数、预测窗口、0 缺失规则、DAG 评估量纲和三类关系的数学定义保持一致。
稀疏遍历改变求和顺序，浮点结果允许小幅数值差异，不能保证逐位相同或完整训练曲线逐轮一致。

## 验证

47 项测试通过，覆盖三个时间编码、多种块大小、多样本、输出及所有参与计算的参数梯度对齐，
以及时间乱序/同时间戳、查询输入零梯度、未来外生时间方向、全空邻域、填充和无边关系的零梯度。
完整 CLI 同时测试 dense 与 sparse_csr。

下述历史基准使用未启用未来外生的旧配置，不代表当前 all 配置的耗时。实际 DIST 单窗口 CPU 基准使用：18,910 节点、hid_dim=16、2 层时间关系、1 层跨变量关系、接收块 128、单线程。
本机一次顺序测试：dense 为 62.45 秒，sparse_csr 为 27.62 秒，约 2.26 倍加速；峰值进程 RSS 分别约 1.15 GB 和 1.05 GB。真实窗口预测最大绝对差为 4.77e-7，参数梯度最大绝对差为 5.44e-7。报告见 `reports/dist_sparse_benchmark.json`。这是单窗口前向+反向，不是全 epoch 或 GPU 实测。
CSR 的索引和原生算子存在额外开销，峰值进程内存不保证低于已经分块的 dense 实现；报告同时给出实测内存。
全部合法边的数量仍可随观测数量平方增长，因此不能承诺 DIST 从此变为线性计算，CPU 全量训练仍可能较慢。

可在目标机器上分开运行以下基准（两次使用相同种子和权重）：

```bash
python scripts/benchmark_exen_sparse.py --backend dense
python scripts/benchmark_exen_sparse.py --backend sparse_csr
```

默认 CPU 单线程。GPU 基准另加 `--device cuda:0`；报告和输出/梯度张量保存在 `results/sparse_benchmark/`。
基准脚本只执行首个训练窗口，不更新参数，不读取测试集计算指标。
