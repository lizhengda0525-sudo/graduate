# TC-cMLP

本项目用于训练 Temporally Coupled component-wise MLP，并从连续时间窗口中提取动态 Granger causal matrix。代码独立编写，不依赖 `Neural-GC-master`。

## 目录

- `configs/`：Synthetic data 与 HUP116 实验参数。
- `src/tc_cmlp/data/`：Synthetic data、BIDS iEEG 和时间窗口处理。
- `src/tc_cmlp/models/`：TC-cMLP 模型与训练过程。
- `src/tc_cmlp/analysis/`：网络特征、SOZ 排名和评价指标。
- `src/tc_cmlp/pipelines/`：完整实验流程。
- `tests/`：数据、模型和评价函数测试。
- `results/`：程序运行后生成的结果，该目录不进入 Git。

## 安装

在项目目录中运行：

```powershell
python -m pip install -e ".[dev]"
```

## Synthetic data 实验

```powershell
python -m tc_cmlp.cli synthetic --config configs/synthetic.yaml
```

输出包括三种方法的 causal matrix、训练记录和 Edge AUPRC、Edge F1、causal strength error。

## HUP116 实验

```powershell
python -m tc_cmlp.cli hup --config configs/hup116.yaml
```

程序读取 HUP116 第一段 ictal EDF，移除 bad channel，完成滤波、降采样、标准化和滑动窗口处理，随后训练指定方法并保存 causal matrix、outflow、channel 排名和 SOZ 评价结果。

## 测试

```powershell
python -m pytest
```

可以使用 `configs/smoke.yaml` 快速检查完整 Synthetic data 流程。

## 方向约定

`causal_matrix[window, target, source]` 表示 `source -> target`。Outflow 对 source 所在列求和，inflow 对 target 所在行求和，所有网络特征均忽略矩阵对角线。
