from pathlib import Path
import csv
import json


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    project = Path(__file__).resolve().parent
    settings = json.loads((project / 'config.json').read_text(encoding='utf-8'))
    result = project / 'results' / settings['run_id']
    scores = read_rows(result / 'tables/selected_summary.csv')
    differences = read_rows(result / 'tables/paired_differences.csv')
    selected = read_rows(result / 'tables/chosen_configurations.csv')
    provenance = json.loads((result / 'audit/provenance.json').read_text(encoding='utf-8'))
    audit = json.loads((result / 'audit/independent_prediction_check.json').read_text(encoding='utf-8'))
    resource = json.loads((result / 'audit/resources.json').read_text(encoding='utf-8'))
    assert audit['passed']
    lines = [
        '# 返回代码阶段后的探索实验结果',
        '',
        '日期：2026-10-29。本轮按用户授权使用原测试数据辅助选参。当前是课程探索实验，不是新的独立受试者泛化测试；原formal_v1与29页初稿保留，本轮尚未重写专利书。',
        '',
        '核心结论：可以将后续模拟专利实施例重点限定为“20 dB高斯仿真噪声下的手机六类稳定活动识别”。新噪声实例检查支持局部收益，但没有选出优于原完整方案的新参数，不能将本轮称为又一次技术改进。',
        '',
        '## 1 调整依据与实际运行',
        '',
        '| 项目 | 本次实际情况 |',
        '|---|---|',
        '| 调整依据 | 原实验在20 dB高斯条件有局部收益；用户允许测试辅助调参，故聚焦该条件搜索K、λ、β |',
        '| 数据用途 | 21人仍只用于统计拟合及分类器训练；原9名测试人及标签用于选参和后续评价 |',
        '| 参数搜索 | 121组候选×3训练种子，共363次分类器拟合；另有5次小规模试运行 |',
        '| 目标 | 20 dB高斯下，对5个噪声种子和3个训练种子平均的受试者平均六类Macro-F1 |',
        '| 选择后检查 | 相同9人、更换5个噪声种子；保留干净、高斯五档、脉冲三档41条件 |',
        f"| 资源 | 已记录处理区间{resource['wall_seconds']:.2f}秒；采样峰值RSS {resource['sampled_peak_rss_bytes']/2**20:.2f} MiB；CPU单线程。该区间到预测汇总结束，未包含随后绘图及独立审计 |",
        f"| 预测保存 | 选参阶段{provenance['tuning_prediction_rows']:,}条；选择后网格{provenance['selected_prediction_rows']:,}条 |",
        f"| 审计 | sklearn复算{audit['tuning_subject_metrics_verified_with_sklearn']:,}个人级分数一致；选择后{audit['diagnostic_prediction_rows_verified']:,}条预测的窗口ID、受试者及真值覆盖一致 |",
        '| 原实验保护 | 原冻结源码哈希核验通过、冻结清单哈希未变；本轮独立目录归档 |',
        '',
        '## 2 选定参数',
        '',
        '| 方法 | K | λ | β | 选参集20 dB分数 % |',
        '|---|---|---|---|---|'
    ]
    for method in settings['methods']:
        row = next(row for row in selected if row['view'] == 'separately_tuned' and row['method'] == method)
        dimension = '66 全特征' if method == 'B0' else row['k']
        lines.append(f"| {method} | {dimension} | {row['lambda']} | {row['beta']} | {float(row['target_score'])*100:.2f} |")
    lines += ['', 'P最终仍为K60、λ0.25、β0.1，与formal_v1完整方案一致。这里证实的是已有方案在明确条件下的收益；新的搜索没有进一步提高该方案。B2也允许搜索β，A1也允许搜索λ，不能只与未调参的弱基线比较。', '', '## 3 相同受试者上使用新噪声种子的结果', '', '下表为受试者平均六类Macro-F1×100。每项平均3训练种子及5个新噪声种子；干净没有噪声重复。不能称为独立测试准确率。', '', '| 条件 | B0 | B1 | B2 | A1 | P |', '|---|---|---|---|---|---|']
    conditions = [('clean', 'clean', '干净'), ('gaussian', '30', '高斯30 dB'), ('gaussian', '20', '高斯20 dB'), ('gaussian', '10', '高斯10 dB'), ('gaussian', '0', '高斯0 dB'), ('gaussian', '-5', '高斯−5 dB'), ('impulse', '20', '脉冲20 dB'), ('impulse', '10', '脉冲10 dB'), ('impulse', '0', '脉冲0 dB')]
    for noise, snr, label in conditions:
        values = []
        for method in settings['methods']:
            row = next(row for row in scores if row['view'] == 'separately_tuned' and row['method'] == method and row['noise_type'] == noise and row['snr_db'] == snr)
            values.append(f"{100*float(row['mean']):.2f}")
        lines.append('| ' + ' | '.join([label, *values]) + ' |')
    lines += ['', '## 4 重点条件的对照和消融', '', '| 视角 | 比较 | 20 dB差值 百分点 | 描述性95%区间 百分点 |', '|---|---|---|---|']
    for view, label in [('separately_tuned', '各自调参'), ('matched_ablation', '同K同权重消融')]:
        for baseline in ['B0', 'B1', 'B2', 'A1']:
            row = next(row for row in differences if row['view'] == view and row['comparison'] == f'P-{baseline}' and row['noise_type'] == 'gaussian' and row['snr_db'] == '20')
            values = [100*float(row[key]) for key in ['mean_difference', 'ci95_lower', 'ci95_upper']]
            lines.append(f'| {label} | P−{baseline} | {values[0]:+.3f} | [{values[1]:+.3f}, {values[2]:+.3f}] |')
    lines += [
        '',
        '同权重消融以P的K60、λ0.25、β0.1为锚：B2去掉稳定性项、A1去掉冗余项、B1同时去掉两项。因此该20 dB条件可用来说明两项联合的实测作用。B0为全66维参照，与其比较的约2.80个百分点收益不能写成相对B2提高2个百分点。',
        '',
        '区间先平均随机重复，再按9个人配对bootstrap 2000次，未校正根据这9人的标签进行候选选择的偏差。新噪声种子没有使这9人成为独立测试人。这里的“区间不跨0”仅为探索描述，不宣称确认性统计显著。',
        '',
        '## 5 后续专利书可以如何聚焦',
        '',
        '| 内容 | 建议写法及边界 |',
        '|---|---|',
        '| 技术区别 | 用训练配对高斯扰动估计尺度化稳定性惩罚，与互信息及相关冗余联合贪心选列；沿用S101—S112，计算步骤未变 |',
        '| 实施例条件 | HAPT、腰部手机六通道、50 Hz、128点窗、六类稳定活动、训练参考20 dB高斯仿真噪声 |',
        '| 有益效果 | 重点使用本轮20 dB结果及同权重消融；明确数据曾参与选参、另用新噪声实例检查 |',
        '| 适用边界 | 干净与较强高斯噪声不属于已证实增益范围；不主张全工况提高，不把仿真噪声称为真实设备噪声实测 |',
        '| 材料分工 | 专利式正文可以重点说明有效实施例；全候选、全网格和无效条件保留在实验附件，便于追溯 |',
        '| 新颖性 | 仍未专利检索；实验有效性不等于已经证实的新颖性、创造性或可授权性 |',
        '',
        '干净条件P仍低于B2约0.527个百分点，强高斯条件也无总体优势。关注有效条件可以缩小论证范围，但不能通过删除条件标签把局部优势写成一般保证。',
        '',
        '## 6 产物与复现',
        '',
        '项目与命令见hapt_exploration_v2/README.md；参数在config.json；所有候选分数在tables/candidate_summary.csv；选定参数在tables/chosen_configurations.csv；表中分数与区间在selected_summary.csv和paired_differences.csv；保存tuning_predictions.npz、selected_predictions.csv.gz、日志、选列轨迹、模型与审计。后续报告应以明确版本引用，不能混合formal_v1与本轮数字。',
        '',
        '本轮没有修改原课程报告初稿，没有对外发送或提交。下一步可依据上述范围重写模拟申请材料中的有效实施例及有益效果。'
    ]
    text = '\n'.join(lines) + '\n'
    (project.parent / '12_探索调参与有效条件结果说明.md').write_text(text, encoding='utf-8')
    (project / 'RESULTS.md').write_text(text, encoding='utf-8')
    print('已生成中文结果说明')


if __name__ == '__main__':
    main()
