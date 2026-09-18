# TC-cMLP

本项目研究时变非线性 Granger 因果学习，并将其用于癫痫动态有效连接分析与 seizure onset zone（SOZ）定位。

## 研究目标

滑动窗口 cMLP 在各时间窗口独立估计因果网络，短窗口条件下容易受到样本数量、脑电噪声与训练波动的影响。本项目研究 TC-cMLP（Temporally Coupled component-wise MLP），联合估计连续时间窗口中的动态因果结构。

## 方法概要

TC-cMLP 为每个时间窗口设置独立的第一层因果参数，并在各窗口之间共享后续非线性网络层：

- `W_1^(w)` 描述第 `w` 个窗口的 Granger 因果结构；
- `W_2:L` 学习跨窗口共享的非线性预测关系；
- group sparsity 用于学习稀疏连接；
- temporal regularization 用于约束相邻窗口的 causal score 变化。

窗口 `w` 中从 source channel `j` 到 target channel `i` 的 causal score 定义为：

$$
C_{j\rightarrow i}^{(w)}=
\left\|W_{i\leftarrow j}^{(1,w)}\right\|_F.
$$

训练目标由预测误差、group sparsity、temporal regularization 和 weight decay 组成：

$$
\mathcal{L}=
\mathcal{L}_{\mathrm{pred}}
+\lambda_s\mathcal{L}_{\mathrm{group}}
+\lambda_t\mathcal{L}_{\mathrm{temporal}}
+\lambda_w\mathcal{L}_{\mathrm{weight}}.
$$

## 验证内容

1. 在具有已知动态因果结构的 synthetic data 上评价 edge detection、causal strength recovery 和 change-point recovery。
2. 与 sliding-window sparse VAR、independent cMLP、fully shared cMLP 和无 temporal regularization 的时变第一层模型比较。
3. 在 HUP iEEG 上分析 dynamic causal outflow、network change 和 SOZ localization。
4. 按 patient、seizure 或 recording 划分数据，然后在各数据集合内生成滑动窗口，防止相邻窗口造成数据泄漏。

## 项目目录

- `DataSet/CHB-MIT/`：CHB-MIT EEG 数据。
- `DataSet/HUP_iEEG/`：HUP iEEG 数据。
- `Paper/`：上一篇 sparse cMLP 研究论文。
- `Result/`：实验输出目录。
- `Code/`：模型代码目录。
- `Codde/Neural-GC-master`: 上一篇 sparse cMLP 研究论文对应的代码。

## 当前状态

项目处于方法实施阶段。当前目录包含研究数据与参考论文，模型代码、实验配置和运行命令尚未加入。
