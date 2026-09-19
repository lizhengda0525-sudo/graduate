# TC-cMLP

本工程中的名称固定为：`cMLP` 表示 Neural-GC 原始方法；`TC-cMLP` 表示在原始 cMLP 上加入跨窗口参数共享和时间约束的方法。S-cMLP 不包含在本工程中。

此前的 TC_cMLP 工程及其结果保存在 `../备份/TC_cMLP_旧版_20260919/`；较早的 Neural-GC 代码副本保存在 `../备份/Neural-GC-master_旧版_20260919/`。本工程只读取 `../Neural-GC-original/` 中的原始代码。

## 原始代码与方法关系

- `../Neural-GC-original/models/cmlp.py` 是原始 cMLP 代码，工程直接导入其中的 `cMLP`、`train_model_ista`、`prox_update`、`regularize` 和 `ridge_regularize`。
- `cmlp` 方法逐窗口调用原始 `cMLP` 和 `train_model_ista`，用于原始方法基线。
- `tc_no_temporal` 与 `tc` 都由原始 `cMLP` 预测网络组成。每个窗口保留独立的第一层；相同目标变量的后续层在窗口之间共享。
- `tc` 对相邻窗口的第一层因果强度加入 total variation 约束；`tc_no_temporal` 将该约束系数设为零。
- 三种方法均使用原始代码的 `GC` 提取因果强度与二值关系，矩阵方向为 `target × source`。

TC-cMLP 的训练目标为各窗口、各目标变量的预测均方误差之和，加上原始 cMLP 的 GL group penalty、共享后续层的 ridge penalty 和相邻窗口因果强度的 total variation penalty。第一层采用与原始代码相同的 GL proximal update；共享后续层的梯度步长按窗口数量平均。

## 工程内容

- `src/tc_cmlp/official.py`：连接官方 Neural-GC 仓库，并核对模块来源。
- `src/tc_cmlp/model.py`：基于原始 cMLP 网络的 TC-cMLP。
- `src/tc_cmlp/training.py`：原始 cMLP 基线和 TC-cMLP 训练。
- `src/tc_cmlp/data/`：Synthetic 与 HUP iEEG 数据。
- `scripts/extract_hup_segment.py`：从 EDF 提取指定时间区间，仅在原始信号区间文件缺失时使用。
- `运行脚本/`：HUP 数据提取、预处理和训练的 PowerShell 入口及说明。
- `src/tc_cmlp/analysis.py`：因果边、变化窗口和 SOZ 评价。
- `src/tc_cmlp/experiment.py`：结果数据、模型参数和运行记录保存。
- `run.py`：直接运行的 Python 入口，显示训练进度、耗时和评价指标。
- `configs/`：Synthetic 与 HUP116 的参数。
- `results/`：运行后生成的结果目录；默认不进入 Git。

## 运行命令

```powershell
py -m pip install -e ".[dev]"
py run.py --task synthetic --config configs/synthetic.yaml --output-dir results/synthetic/new_run
py -m pytest
```

HUP116 的预处理窗口已经保存在 `results/hup/preprocessed/HUP116_run1_windows.npz`。直接运行训练：

```powershell
py run.py --task hup --config configs/hup116.yaml --output-dir results/hup/HUP116_run1_new
```

终端会显示当前方法、窗口或迭代进度、目标函数、已用时间，以及每种方法的 SOZ Precision、Sensitivity、F1 和训练秒数。原始 cMLP 每完成 10 个窗口显示一次；训练过程中每 30 秒显示一次当前状态。可以用 `--progress-every` 和 `--heartbeat-seconds` 调整这两个间隔。指定新的 `--output-dir` 可保留已有结果；结果目录必须位于工程的 `results/` 下。原始信号区间保存在 `results/hup/cache/HUP116_run1_segment.npz`。信号提取及预处理命令见[运行脚本说明](./运行脚本/说明.md)。提取信号需要 `pyedflib`，可安装 `.[preprocess]`。

每次实验在指定的 `results/` 子目录保存配置、原始仓库 commit、训练记录、模型参数、因果强度矩阵、二值因果矩阵和评价指标。`training_times.json` 分别记录每种方法的训练秒数和起止时间；训练秒数不包含结果保存与评价计算。目录已存在时程序会停止，防止覆盖原有结果。

Synthetic 功能检查使用 `configs/synthetic_smoke.yaml`；完整的 seed 42 结果保存在 `results/synthetic/seed_42/`；训练耗时比较使用 `configs/synthetic_timing_seed_42.yaml`，结果保存在 `results/synthetic/seed_42_timing/`。HUP116 的完整实验仍待运行。
