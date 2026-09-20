# 项目长期记忆（G:\graduate）

## 论文写作约定

- TC-cMLP 新论文以独立新工作形式撰写：正文不引用作者此前的 iCONECCT 会议论文，不出现 S-cMLP 等前作方法名与前作实验数值；前作 PDF（`Paper/` 下）仅作为段落组织与图表叙述的写法参考。
- 论文为英文 IEEE 会议格式；中文规划文档在 `论文内容/`，公式与实现以 `Code/TC_cMLP/src/tc_cmlp/` 实际代码为准。
- SOZ 判定规则：逐窗口 outflow 排名前 18% 为异常 channel，ictal windows 内计数，最终取前 18% 为预测 SOZ channel。

## 环境备注

- pip 默认镜像（tsinghua）在该机器上解析失败，安装包需加 `-i https://pypi.org/simple`。
- Bash 工具的 PATH 缺少 dirname/tail 等基础命令，长命令建议直接运行 Python 脚本文件。
