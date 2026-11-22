# 手机惯性信号活动识别：可复现实验项目

这是《数据分析与特征提取》课程的技术实验项目，不是最终课程报告，也不构成专利新颖性或可专利性结论。Python代码按已确定的A题技术方案实现，CPU单线程运行。

## 1. 项目结构

| 路径 | 职责 |
|---|---|
| `run.py` | 命令行入口、配置加载、线程及缓存目录设置 |
| `configs/protocol_v1.json` | 原定科学协议：数据划分、噪声、参数网格、种子、评价规则 |
| `configs/formal.json` | 实现层配置：路径、数值下限、试运行范围、图形及计时设置 |
| `hapt/data.py` | 官方数据下载与校验、预处理、记录级划分、窗口生成、训练参考功率 |
| `hapt/features.py` | 66维时频/相关特征、连续记录加噪、共享条件缓存 |
| `hapt/selection.py` | 训练内互信息、扰动稳定性、冗余估计与贪心筛选 |
| `hapt/metrics.py` | 六类指标、受试者平均Macro-F1及混淆矩阵 |
| `hapt/experiment.py` | 试运行、三折验证、配置选择、重拟合、模型冻结、最终评价与计时 |
| `hapt/verification.py` | 已知信号、指标对照、噪声一致性、边界与划分隔离检查 |
| `hapt/plotting.py` | 按人汇总、配对bootstrap、绘图源数据、PNG/SVG图表 |
| `hapt/common.py` | 哈希、表格/JSON、日志、实际进程RSS监测 |
| `requirements-lock.txt` | 实际核验环境的锁定依赖 |
| `results/formal_v1/` | 正式实验配置、审计、模型、预测、指标和图表 |

完整公式及实验设计见`docs/`中的04—07阶段设计存档，工作区`outputs`中也保留原文件；当前实际结果以`RESULTS.md`为准。配置采取“原协议＋实现层”合并，实际合并结果保存为`configuration_snapshot.json`。没有利用最终测试结果改K、λ、分类器或噪声条件。

## 2. 环境与依赖

本机实际使用工作目录的`work/.venv/Scripts/python.exe`，Python 3.12；依赖版本以`requirements-lock.txt`及结果中的资源记录为准。无GPU需求，不依赖pandas或联网服务进行训练。

在本聊天工作目录运行PowerShell：

```powershell
$ExperimentPython = '.\work\.venv\Scripts\python.exe'
& $ExperimentPython -m pip check
```

如果在另一台电脑重建，可先使用已安装的Python 3.12创建虚拟环境，再安装锁定依赖：

```powershell
py -3.12 -m venv .\work\.venv
$ExperimentPython = '.\work\.venv\Scripts\python.exe'
& $ExperimentPython -m pip install -r .\outputs\hapt_project\requirements-lock.txt
```

本环境受限沙箱曾阻止科学计算动态库加载；同一虚拟环境在获准的普通桌面权限下可正常运行。不要把该系统限制误判为模型失败。入口把线程数设为1，并将Matplotlib配置缓存写入`work/matplotlib-cache`。

## 3. 数据与许可

官方来源：[UCI HAPT，ID 341](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions)。使用RawData连续六通道真实公开实验记录；噪声为程序模拟。未使用包内561维预提取数据。

官方ZIP地址、SHA256、物理单位和许可证据在配置与`results/formal_v1/data`中记录。官网标注CC BY 4.0，原README另有非商业限制；本项目限定署名的非商业课程研究，保存原README并注明自行分窗、特征提取和加噪。数据研究参考题名为Reyes-Ortiz等人的“Transition-Aware Human Activity Recognition Using Smartphones”（2015）；完整出版元信息未在本项目中编造。

`prepare`优先使用`work/hapt-341.zip`。文件不存在时从配置中的UCI官方HTTPS链接下载；任何情况下均检查SHA256。原始ZIP及缓存只放`work`，不向外上传。结果包内提供数据清单及复现获取方法，不重复打包公开原始数据。

## 4. 从数据到结果的命令

在本聊天工作目录执行，默认工作区自动按`outputs/hapt_project`位置推导。若项目放在其他位置，使用`--workspace`指定含`work`和`outputs`的目录。

```powershell
$ExperimentPython = '.\work\.venv\Scripts\python.exe'
$ExperimentEntry = '.\outputs\hapt_project\run.py'
& $ExperimentPython $ExperimentEntry prepare
& $ExperimentPython $ExperimentEntry check
& $ExperimentPython $ExperimentEntry smoke
& $ExperimentPython $ExperimentEntry train
& $ExperimentPython $ExperimentEntry evaluate
& $ExperimentPython $ExperimentEntry plot
```

也可在全新工作区顺序执行：

```powershell
& $ExperimentPython $ExperimentEntry all
```

| 命令 | 行为 |
|---|---|
| `prepare` | 核验ZIP、逐记录检查、先分人/记录再分窗、输出清单 |
| `check` | 用已知解析信号和参考指标检查实现，确认原始采集单元隔离 |
| `smoke` | 只用开发集1/3/5号训练、6号验证，跑通全部5种方法；中间结果在`work` |
| `train` | 要求检查和试运行已通过；完整三折/三种子验证，重拟合并冻结模型 |
| `evaluate` | 先验证源码/配置/模型哈希，只评价冻结模型；保存全部预设条件预测与指标，不fit |
| `plot` | 从已有预测指标汇总并生成图表；不重训或调参 |

如果已经存在冻结训练，`train`只验证哈希，不重新选参。若最终评价已完成，`evaluate`不重新开启测试评分，只保留并校验已有结果；计时基准可能单独重测。需要完全从头复现时，应在新的工作目录使用相同源码和配置，或者复制`configs/formal.json`为新配置，将`implementation.run_id`、`scratch_root`、`result_relative_path`分别改为新的运行标识和目录，再通过`--config`传入。不要覆盖已有冻结结果或为提高测试分数修改科学参数。

## 5. 实验规模与评价含义

21名开发受试者、9名测试受试者；记录内128点窗、步长64，保守排除标注边界10点，预计6928/2790窗。开发集按人三折。标准化、训练参考功率、常数掩码、互信息、稳定性、冗余和选列均只fit当前训练折；最终重拟合仅使用完整21人。

五组：B0全有效特征；B1互信息；B2互信息＋冗余；A1互信息＋稳定性；P完整方案。K=20/40/60，λ=0.25/0.5/1，β=0.1，C=1。另保留λ=0.5固定消融。分类器不做方法私有调参，不用加噪样本增强主分类器。

最终测试包括干净、高斯30/20/10/0/−5 dB、脉冲20/10/0 dB，每噪声条件5个seed，训练重复3个seed。SNR相对训练记录中心化参考功率定义；另保存逐记录实现SNR。噪声在连续记录上添加，重叠窗共享原始采样噪声，所有方法共享同一条件。不能称为真实设备噪声工况实测。

主指标为9个人分别计算六类Macro-F1后等权平均；不要与全部窗口池化Macro-F1混用。95%区间先平均训练/噪声重复，再按9个人配对bootstrap 2000次；种子和重叠窗口不是独立受试者。图表使用英文规范标注，颜色同时配合不同点形。误差条定义、数据路径与变换均保存于`figures/figure_provenance.json`。

## 6. 结果文件与复核

| 路径 | 可复核内容 |
|---|---|
| `data/*manifest*`、`data_audit.json` | 记录/窗口来源、标签区间、集合隔离 |
| `audit/checks.json`、`audit/smoke_complete.json` | 检查与仅开发集试运行证据 |
| `tables/cv_*` | 所有候选与折/seed结果、特征选择稳定性 |
| `fit_statistics/` | 各折及最终训练的MI、稳定性、相关矩阵、训练来源 |
| `models/*.json`与`*.joblib` | 参数、所选特征、评分轨迹、拟合对象；只加载本项目可信本地模型 |
| `freeze_manifest.json` | 源码/配置/模型哈希及冻结的评价对象 |
| `predictions/*.csv.gz` | 每条件每模型每窗口的真值与预测；窗口ID可连接数据清单 |
| `tables/test_subject_metrics.csv` | 每个受试者、训练seed、噪声seed与条件的原始指标 |
| `tables/test_overall_metrics.csv` | 池化指标与受试者平均指标，含各次预测耗时 |
| `tables/figure_source_scores.csv` | 所有作图分数、95%区间、随机重复波动、各类召回 |
| `tables/paired_differences.csv` | 同预算等视角的方法差值及配对区间 |
| `tables/gaussian_grid_summary.csv` | 五个预设高斯条件等权平均，不能当实际工况概率 |
| `tables/realized_snr.csv` | 原始记录上事后测得SNR，不用于自适应调噪 |
| `tables/inference_benchmark.csv` | 开发集窗口、单线程、预热后5次端到端计时 |
| `audit/resources_*.json` | 真实阶段耗时、采样进程RSS、环境版本 |
| `figures/` | PNG、SVG、图表源数据路径与误差条说明 |

本次已完成正式运行与独立预测审计，完整结果解读见项目根目录`RESULTS.md`。主要发现是完整方案仅在部分条件下有效，五级高斯平均表现低于最接近的B2基线；无效、局部有效及强噪声失败情况全部保留。源码与结果均可审阅；不发送邮件、不对外提交课程材料。

## 7. 软件流程致谢

实现与图形审查参考了Scientific Agent Skills中的scikit-learn和scientific-visualization工作流。按技能要求，已于2026-10-28核验arXiv元信息并记录：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner. *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. 2026. [arXiv:2609.00065](https://arxiv.org/abs/2609.00065)，[DOI](https://doi.org/10.48550/arXiv.2609.00065)。该引用说明软件工作流来源，不是本方案技术新颖性的证据。
