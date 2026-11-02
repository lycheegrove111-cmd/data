# Glaucoma 数据清洗结果

本目录保存数据核验与清洗脚本及其结果。原始图片和 CSV 不复制到 Git 仓库。

运行：

```powershell
C:\Users\26235\.conda\envs\pytorch\python.exe data_cleaning.py `
  --data-root E:\比赛们\大数据挑战\glaucoma_first\glaucoma_first `
  --output-root results
```

清洗规则：保留官方 `train/valid/test` 划分；删除缺失或无法读取的图片、重复行和非法标签；保留年龄异常值用于追踪；缺失年龄使用对应划分的中位数填充。原始数据不被修改。

最佳模型权重（本机路径）：

```text
E:\glaucoma\glaucoma_training\runs\baseline\best.pt
```

模型文件未复制到本仓库；在其他机器上运行评估前，需要提供对应的权重文件。
