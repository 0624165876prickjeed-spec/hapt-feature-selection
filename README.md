# 手机惯性信号活动识别课程设计


本项目比较全特征、互信息筛选和加入稳定性、冗余约束的特征选择方法。报告采用target20_v2实验结果。

| 文件或目录 | 内容 |
|---|---|
| 根目录DOCX、PDF | 课程设计说明书，含任务书及空白评分页 |
| outputs/hapt_exploration_v2 | 参数搜索、对比实验、消融实验及结果 |
| outputs/hapt_project | 前期实验代码及训练统计量，用于重建完整计算过程 |
| report_figures | 报告中的四张附图 |
| data_sample | 24个真实数据窗口、六通道信号、66维特征及字段说明 |
| 开源网址.txt | scikit-learn公开源码及文档网址 |
| requirements-lock.txt | 实验使用的Python依赖版本 |
| reproduce.py、verify_main.py | 复现入口及主结果重新训练检查 |

## 环境准备

使用Python 3.12、CPU单线程，16 GB内存即可运行。以下PowerShell命令均在解压后的提交包根目录执行：

```powershell
py -3.12 -m venv .\work\.venv
$CoursePython = '.\work\.venv\Scripts\python.exe'
& $CoursePython -m pip install -r .\requirements-lock.txt
& $CoursePython -m pip check
```

## 数据准备

采用[UCI HAPT数据集](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions)的RawData连续记录。官网标注CC BY 4.0，原数据包README还注明署名及非商业使用要求。本项目用于非商业课程研究，原说明保存在结果目录data/SOURCE_README.txt。

提交包的data_sample提供24个真实窗口的整理样本，覆盖两个分区与六类活动，并附原始信号、窗口索引及66维特征。它用于展示整理效果，完整实验仍需以下原始数据包。

原始数据包约76 MB，不在提交包中重复提供。可按下列命令下载；未手动下载时，实验程序也会从同一官方地址获取，并检查文件完整性。

```powershell
New-Item -ItemType Directory -Force .\work | Out-Null
Invoke-WebRequest -Uri 'https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip' -OutFile .\work\hapt-341.zip
```

| 项目 | 说明 |
|---|---|
| 原始记录 | 30名受试者、61条记录，加速度及角速度各三轴，50 Hz |
| 原始文件 | RawData/acc_expXX_userYY.txt及对应gyro文件；labels.txt提供活动区间 |
| 物理单位 | 加速度g，角速度rad/s |
| 训练部分 | 21人、42条记录、6928窗 |
| 调参评价部分 | 9人、19条记录、2790窗 |
| 分窗 | 128点窗、64点步长；先按人和记录划分，再生成窗口 |

仅保留六类稳定活动，排除姿态转换及标注边界。噪声由程序添加，为仿真噪声。数据来自腰部手机实验，不代表其他设备或佩戴位置。

## 运行方法

```powershell
& $CoursePython .\reproduce.py smoke
& $CoursePython .\reproduce.py verify-main
& $CoursePython .\reproduce.py audit
```

| 命令 | 作用 |
|---|---|
| smoke | 检查特征、指标、分窗边界和划分，再用小规模配置跑通五种方法 |
| verify-main | 按选定参数训练15个模型，重新生成5组20 dB高斯噪声并核对主结果 |
| audit | 从已保存预测重新计算指标，无需下载数据或重新训练 |
| formal | 自动试运行，再执行完整参数搜索和压力条件评价 |
| rebuild-all | 从原始数据重建前期训练统计，再执行后续完整实验 |

完整实验执行`& $CoursePython .\reproduce.py formal`。中间文件和新结果保存在work中，不覆盖随包结果。同名运行目录已存在时程序停止，可移走原目录后重新执行。formal复用前期21人训练统计，与报告中的实际计算过程一致；从头计算时执行rebuild-all。首次提取特征的耗时会高于报告中的缓存运行时间。

## 程序与结果

核心模块位于outputs/hapt_exploration_v2/core/hapt，实验入口为run_exploration.py。参数集中在config.json及结果目录configuration_snapshot.json。

| 模块 | 功能 |
|---|---|
| data.py | 下载、读取、划分、窗口索引和训练参考功率 |
| features.py | 连续记录加噪、66维时频及轴间相关特征 |
| selection.py | 互信息、扰动稳定性和冗余约束的计算与选列 |
| experiment.py | 标准化、LinearSVC训练及评价 |
| metrics.py | Macro-F1、准确率、各类召回及混淆矩阵 |
| plotting.py及入口绘图函数 | 汇总结果和绘制图表 |
| verification.py | 已知信号、指标及边界条件检查 |

主要结果位于outputs/hapt_exploration_v2/results/target20_v2：tables存放候选与汇总指标，models存放模型及选列结果，figures存放图表，两个prediction文件保存逐窗口预测。报告图2、图3、图4的数据分别来自candidate_summary.csv、selected_summary.csv、paired_differences.csv。

20 dB高斯噪声下，完整方案的受试者平均六类Macro-F1为90.8661%，B2为90.1693%，相差0.6968个百分点。去掉稳定性项或冗余项，分别下降约0.699和0.710个百分点。干净及更强高斯噪声下没有普遍改善，完整结果保留在报告附录和CSV中。

9名评价受试者参与参数选择，新噪声评价仍使用这9人。训练统计与分类器只使用21人数据。95%区间按9人配对bootstrap 2000次计算，未校正选参偏差。结果不能作为独立受试者泛化或专利新颖性的证明。
