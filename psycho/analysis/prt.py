from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl
from plotly.subplots import make_subplots

from psycho.analysis.utils import extract_trials_by_block

# ==================== 1. 数据加载和预处理 ====================
print("=" * 60)
print("PRT数据分析")
print("=" * 60)

# 读取CSV文件

file_path = Path(input("请输入PRT数据文件路径:\n").strip("'").strip()).resolve()
df = pl.read_csv(file_path)
# 查看数据基本信息
print("数据基本信息:")
print(f"数据行数: {df.height}")
print(f"数据列数: {df.width}")
print(f"Block数量: {df['block_index'].fill_null(-1).n_unique() - 1}")
# 查看数据基本信息
print("\n列名:")
print(df.columns)

trials_df = extract_trials_by_block(df, target_block_indices=[0, 1, 2])

print(f"提取的试次数: {trials_df.height}")
print(f"包含的列: {trials_df.columns}")

# 添加分析需要的列
trials_df = trials_df.with_columns(
    [
        # 标记选择是否正确
        (pl.col("stim") == pl.col("choice")).alias("correct"),
        # 标记是否获得奖励（3或9）
        pl.col("reward").gt(0).alias("rewarded"),
        # 标记是否错误（-1）
        (pl.col("reward") == -1).alias("error"),
        # 添加trial_in_block（已经由工具函数添加）
    ]
)

# ==================== 2. 识别每个Block的Rich刺激 ====================
print("\n" + "=" * 60)
print("识别每个Block的Rich刺激")
print("=" * 60)

# 计算每个Block中每种刺激获得的奖励次数
rich_stim_results = {}

for block in sorted(trials_df["block_index"].unique()):
    block_data = trials_df.filter(pl.col("block_index") == block)

    # 统计s刺激正确且获得奖励的次数
    s_rewards = block_data.filter(
        (pl.col("stim") == "s") & (pl.col("correct")) & (pl.col("rewarded"))
    ).height

    # 统计l刺激正确且获得奖励的次数
    l_rewards = block_data.filter(
        (pl.col("stim") == "l") & (pl.col("correct")) & (pl.col("rewarded"))
    ).height

    # 确定rich刺激（奖励次数多的）
    rich_stim = "s" if s_rewards > l_rewards else "l"
    lean_stim = "l" if rich_stim == "s" else "s"

    rich_stim_results[block] = {
        "rich_stim": rich_stim,
        "lean_stim": lean_stim,
        "s_rewards": s_rewards,
        "l_rewards": l_rewards,
        "total_trials": block_data.height,
    }

    print(f"Block {block}:")
    print(f"  s刺激奖励次数: {s_rewards}")
    print(f"  l刺激奖励次数: {l_rewards}")
    print(f"  Rich刺激: '{rich_stim}' (奖励次数更多)")
    print(f"  Lean刺激: '{lean_stim}'")
    print(f"  总试次数: {block_data.height}")

# ==================== 3. 计算SDT指标（log b和log d） ====================
print("\n" + "=" * 60)
print("计算SDT指标（反应偏向和辨别力）")
print("=" * 60)


def calculate_sdt_metrics(data: pl.DataFrame, rich_stim: str) -> dict:
    """计算信号检测理论指标"""
    lean_stim = "l" if rich_stim == "s" else "s"

    # 提取四类试次
    rich_hit = data.filter((pl.col("stim") == rich_stim) & (pl.col("correct"))).height

    rich_miss = data.filter((pl.col("stim") == rich_stim) & (~pl.col("correct"))).height

    lean_hit = data.filter((pl.col("stim") == lean_stim) & (pl.col("correct"))).height

    lean_miss = data.filter((pl.col("stim") == lean_stim) & (~pl.col("correct"))).height

    # Hautus校正：每个单元格加0.5
    rich_hit_c = rich_hit + 0.5
    rich_miss_c = rich_miss + 0.5
    lean_hit_c = lean_hit + 0.5
    lean_miss_c = lean_miss + 0.5

    # 计算log b（反应偏向）
    if (rich_miss_c * lean_hit_c) > 0:
        log_b = 0.5 * np.log10((rich_hit_c * lean_miss_c) / (rich_miss_c * lean_hit_c))
    else:
        log_b = 0.0

    # 计算log d（辨别力）
    if (rich_miss_c * lean_miss_c) > 0:
        log_d = 0.5 * np.log10((rich_hit_c * lean_hit_c) / (rich_miss_c * lean_miss_c))
    else:
        log_d = 0.0

    # 计算击中率
    rich_total = rich_hit + rich_miss
    lean_total = lean_hit + lean_miss

    rich_hit_rate = rich_hit / rich_total if rich_total > 0 else 0
    lean_hit_rate = lean_hit / lean_total if lean_total > 0 else 0

    # 计算漏报率
    rich_miss_rate = 1 - rich_hit_rate
    lean_miss_rate = 1 - lean_hit_rate

    # 计算额外指标
    total_correct = rich_hit + lean_hit
    total_trials = rich_total + lean_total
    overall_accuracy = total_correct / total_trials if total_trials > 0 else 0

    return {
        "log_b": log_b,
        "log_d": log_d,
        "rich_hit_rate": rich_hit_rate,
        "lean_hit_rate": lean_hit_rate,
        "rich_miss_rate": rich_miss_rate,
        "lean_miss_rate": lean_miss_rate,
        "rich_hit": rich_hit,
        "rich_miss": rich_miss,
        "lean_hit": lean_hit,
        "lean_miss": lean_miss,
        "overall_accuracy": overall_accuracy,
        "hit_rate_difference": rich_hit_rate - lean_hit_rate,
    }


# 为每个Block计算SDT指标
sdt_results = {}

for block in sorted(trials_df["block_index"].unique()):
    block_data = trials_df.filter(pl.col("block_index") == block)
    rich_stim = rich_stim_results[block]["rich_stim"]

    results = calculate_sdt_metrics(block_data, rich_stim)
    sdt_results[block] = results

    print(f"Block {block}:")
    print(f"  log_b (反应偏向) = {results['log_b']:.3f}")
    print(f"  log_d (辨别力) = {results['log_d']:.3f}")
    print(f"  Rich刺激击中率 = {results['rich_hit_rate']:.3f}")
    print(f"  Lean刺激击中率 = {results['lean_hit_rate']:.3f}")
    print(f"  击中率差异(Rich-Lean) = {results['hit_rate_difference']:.3f}")
    print(f"  总体准确率 = {results['overall_accuracy']:.3f}")

# ==================== 4. 概率分析（关键分析） ====================
print("\n" + "=" * 60)
print("概率分析（论文图3的关键分析）")
print("=" * 60)


def calculate_probability_analysis(
    block_data: pl.DataFrame, rich_stim: str, lean_stim: str
) -> dict:
    """进行概率分析，计算特定条件下的漏报概率"""
    # 确保数据按试次顺序排列
    block_data = block_data.sort("trial_in_block")

    # 添加上一试次的信息
    block_data = block_data.with_columns(
        [
            pl.col("stim").shift(1).alias("prev_stim"),
            pl.col("rewarded").shift(1).alias("prev_rewarded"),
            pl.col("correct").shift(1).alias("prev_correct"),
            pl.col("choice").shift(1).alias("prev_choice"),
        ]
    )

    # 只考虑前一试次正确的情况
    valid_data = block_data.filter(pl.col("prev_correct") == True)

    # 情况A: 分析lean miss概率
    lean_trials = valid_data.filter(pl.col("stim") == lean_stim)

    # A1: 前一个试次是rich且获得奖励
    cond1 = lean_trials.filter(
        (pl.col("prev_stim") == rich_stim) & (pl.col("prev_rewarded") == True)
    )

    # A2: 前一个试次是rich但无奖励
    cond2 = lean_trials.filter(
        (pl.col("prev_stim") == rich_stim) & (pl.col("prev_rewarded") == False)
    )

    # 计算lean miss概率（被试错误选择rich）
    lean_miss_prob1 = (
        (cond1.filter(pl.col("correct") == False).height / cond1.height)
        if cond1.height > 0
        else 0
    )
    lean_miss_prob2 = (
        (cond2.filter(pl.col("correct") == False).height / cond2.height)
        if cond2.height > 0
        else 0
    )

    # 情况B: 分析rich miss概率
    rich_trials = valid_data.filter(pl.col("stim") == rich_stim)

    # B1: 前一个试次是rich且获得奖励
    cond3 = rich_trials.filter(
        (pl.col("prev_stim") == rich_stim) & (pl.col("prev_rewarded") == True)
    )

    # B2: 前一个试次是lean且获得奖励
    cond4 = rich_trials.filter(
        (pl.col("prev_stim") == lean_stim) & (pl.col("prev_rewarded") == True)
    )

    # 计算rich miss概率（被试错误选择lean）
    rich_miss_prob1 = (
        (cond3.filter(pl.col("correct") == False).height / cond3.height)
        if cond3.height > 0
        else 0
    )
    rich_miss_prob2 = (
        (cond4.filter(pl.col("correct") == False).height / cond4.height)
        if cond4.height > 0
        else 0
    )

    # 额外计算：前试次是lean且无奖励的情况
    cond5 = rich_trials.filter(
        (pl.col("prev_stim") == lean_stim) & (pl.col("prev_rewarded") == False)
    )
    rich_miss_prob3 = (
        (cond5.filter(pl.col("correct") == False).height / cond5.height)
        if cond5.height > 0
        else 0
    )

    return {
        "lean_miss_after_rewarded_rich": lean_miss_prob1,
        "lean_miss_after_nonrewarded_rich": lean_miss_prob2,
        "rich_miss_after_rewarded_rich": rich_miss_prob1,
        "rich_miss_after_rewarded_lean": rich_miss_prob2,
        "rich_miss_after_nonrewarded_lean": rich_miss_prob3,
        "counts": {
            "cond1": cond1.height,
            "cond2": cond2.height,
            "cond3": cond3.height,
            "cond4": cond4.height,
            "cond5": cond5.height,
        },
    }


# 执行概率分析
prob_results = {}

for block in sorted(trials_df["block_index"].unique()):
    block_data = trials_df.filter(pl.col("block_index") == block)
    rich_stim = rich_stim_results[block]["rich_stim"]
    lean_stim = rich_stim_results[block]["lean_stim"]

    results = calculate_probability_analysis(block_data, rich_stim, lean_stim)
    prob_results[block] = results

# 打印概率分析结果
for block, results in prob_results.items():
    print(f"\nBlock {block}:")
    print(
        f"  1. Lean miss概率（前试次富刺激有奖励）: {results['lean_miss_after_rewarded_rich']:.3f} (n={results['counts']['cond1']})"
    )
    print(
        f"  2. Lean miss概率（前试次富刺激无奖励）: {results['lean_miss_after_nonrewarded_rich']:.3f} (n={results['counts']['cond2']})"
    )
    print(
        f"  差异（1-2）: {results['lean_miss_after_rewarded_rich'] - results['lean_miss_after_nonrewarded_rich']:.3f}"
    )
    print(
        f"  3. Rich miss概率（前试次富刺激有奖励）: {results['rich_miss_after_rewarded_rich']:.3f} (n={results['counts']['cond3']})"
    )
    print(
        f"  4. Rich miss概率（前试次贫刺激有奖励）: {results['rich_miss_after_rewarded_lean']:.3f} (n={results['counts']['cond4']})"
    )
    print(
        f"  差异（4-3）: {results['rich_miss_after_rewarded_lean'] - results['rich_miss_after_rewarded_rich']:.3f}"
    )

# ==================== 5. 反应时分析 ====================
print("\n" + "=" * 60)
print("反应时分析")
print("=" * 60)

# 计算总体反应时
mean_rt = trials_df["rt"].mean()
median_rt = trials_df["rt"].median()
std_rt = trials_df["rt"].std()

print(f"总体反应时:")
print(f"  均值: {mean_rt:.3f}秒")
print(f"  中位数: {median_rt:.3f}秒")
print(f"  标准差: {std_rt:.3f}秒")

# 按Block和刺激类型分析反应时
rt_by_block = {}

for block in sorted(trials_df["block_index"].unique()):
    block_data = trials_df.filter(pl.col("block_index") == block)
    rich_stim = rich_stim_results[block]["rich_stim"]

    # Rich刺激的反应时
    rt_rich = block_data.filter(pl.col("stim") == rich_stim)["rt"].mean()

    # Lean刺激的反应时
    rt_lean = block_data.filter(pl.col("stim") != rich_stim)["rt"].mean()

    # 正确和错误试次的反应时
    rt_correct = block_data.filter(pl.col("correct"))["rt"].mean()
    rt_error = block_data.filter(~pl.col("correct"))["rt"].mean()

    rt_by_block[block] = {
        "rt_rich": rt_rich,
        "rt_lean": rt_lean,
        "rt_diff": rt_lean - rt_rich,
        "rt_correct": rt_correct,
        "rt_error": rt_error,
    }

    print(f"\nBlock {block}:")
    print(f"  Rich刺激平均RT: {rt_rich:.3f}秒")
    print(f"  Lean刺激平均RT: {rt_lean:.3f}秒")
    print(f"  差异（Lean-Rich）: {rt_lean - rt_rich:.3f}秒")
    print(f"  正确试次平均RT: {rt_correct:.3f}秒")
    print(f"  错误试次平均RT: {rt_error:.3f}秒")

# ==================== 6. 性能趋势分析 ====================
print("\n" + "=" * 60)
print("性能趋势分析")
print("=" * 60)


def analyze_performance_trends(trials_df: pl.DataFrame) -> dict:
    """分析性能随时间和试次的变化趋势"""

    # 按block和试次计算滑动窗口性能
    results = {}

    for block in sorted(trials_df["block_index"].unique()):
        block_data = trials_df.filter(pl.col("block_index") == block).sort(
            "trial_in_block"
        )

        # 计算滑动窗口准确率（窗口大小=10个试次）
        window_size = 10
        window_accuracies = []

        for i in range(0, block_data.height - window_size + 1, 5):  # 步长为5
            window = block_data.slice(i, window_size)
            accuracy = window.filter(pl.col("correct")).height / window_size
            window_accuracies.append(
                {
                    "start_trial": i,
                    "end_trial": i + window_size - 1,
                    "accuracy": accuracy,
                    "midpoint": i + window_size / 2,
                }
            )

        # 计算学习曲线：前1/3 vs 后1/3试次
        total_trials = block_data.height
        third = total_trials // 3

        early_trials = block_data.slice(0, third)
        late_trials = block_data.slice(total_trials - third, third)

        early_accuracy = (
            early_trials.filter(pl.col("correct")).height / third if third > 0 else 0
        )
        late_accuracy = (
            late_trials.filter(pl.col("correct")).height / third if third > 0 else 0
        )

        # 反应时变化
        early_rt = early_trials["rt"].mean()
        late_rt = late_trials["rt"].mean()

        results[block] = {
            "window_accuracies": window_accuracies,
            "early_accuracy": early_accuracy,
            "late_accuracy": late_accuracy,
            "accuracy_change": late_accuracy - early_accuracy,
            "early_rt": early_rt,
            "late_rt": late_rt,
            "rt_change": late_rt - early_rt,
        }

        print(f"\nBlock {block}学习趋势:")
        print(f"  早期准确率(试次1-{third}): {early_accuracy:.3f}")
        print(
            f"  晚期准确率(试次{total_trials - third + 1}-{total_trials}): {late_accuracy:.3f}"
        )
        print(f"  准确率变化: {late_accuracy - early_accuracy:.3f}")
        print(f"  早期RT: {early_rt:.3f}秒")
        print(f"  晚期RT: {late_rt:.3f}秒")
        print(f"  RT变化: {late_rt - early_rt:.3f}秒")

    return results


# 执行趋势分析
trend_results = analyze_performance_trends(trials_df)

# ==================== 7. 使用Plotly创建可视化 ====================
print("\n" + "=" * 60)
print("创建可视化图表")
print("=" * 60)

# 创建子图
fig = make_subplots(
    rows=3,
    cols=3,
    subplot_titles=(
        "1. 反应偏向(Log b)变化",
        "2. 击中率对比",
        "3. 准确率趋势",
        "4. Lean miss概率分析",
        "5. Rich miss概率分析",
        "6. 反应时对比",
        "7. 反应时分布",
        "8. 学习曲线",
        "9. 奖励整合指数",
    ),
    specs=[
        [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
        [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
        [{"type": "histogram"}, {"type": "scatter"}, {"type": "bar"}],
    ],
    vertical_spacing=0.1,
    horizontal_spacing=0.15,
)

# 图1: 反应偏向(Log b)随Block变化
blocks = sorted(sdt_results.keys())
log_b_values = [sdt_results[b]["log_b"] for b in blocks]

# 添加文献中的参考线
md_reference = [0.10, 0.12, 0.15]
control_reference = [0.19, 0.20, 0.21]

fig.add_trace(
    go.Scatter(
        x=blocks,
        y=log_b_values,
        mode="lines+markers+text",
        name="当前被试",
        line=dict(width=3, color="blue"),
        marker=dict(size=12),
        text=[f"{val:.3f}" for val in log_b_values],
        textposition="top center",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=blocks,
        y=md_reference,
        mode="lines",
        name="文献MDD组",
        line=dict(width=2, color="red", dash="dash"),
        opacity=0.7,
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=blocks,
        y=control_reference,
        mode="lines",
        name="文献对照组",
        line=dict(width=2, color="green", dash="dash"),
        opacity=0.7,
    ),
    row=1,
    col=1,
)

# 图2: 击中率对比
rich_hit_rates = [sdt_results[b]["rich_hit_rate"] for b in blocks]
lean_hit_rates = [sdt_results[b]["lean_hit_rate"] for b in blocks]

x_positions = np.arange(len(blocks))

fig.add_trace(
    go.Bar(
        x=x_positions - 0.2,
        y=rich_hit_rates,
        name="Rich刺激",
        marker_color="lightgreen",
        text=[f"{val:.3f}" for val in rich_hit_rates],
        textposition="outside",
    ),
    row=1,
    col=2,
)

fig.add_trace(
    go.Bar(
        x=x_positions + 0.2,
        y=lean_hit_rates,
        name="Lean刺激",
        marker_color="lightcoral",
        text=[f"{val:.3f}" for val in lean_hit_rates],
        textposition="outside",
    ),
    row=1,
    col=2,
)

fig.update_xaxes(
    ticktext=[f"Block {b}" for b in blocks], tickvals=x_positions, row=1, col=2
)

# 图3: 准确率趋势
fig.add_trace(
    go.Scatter(
        x=blocks,
        y=[sdt_results[b]["overall_accuracy"] for b in blocks],
        mode="lines+markers",
        name="总体准确率",
        line=dict(width=3, color="purple"),
    ),
    row=1,
    col=3,
)

# 图4: Lean miss概率分析
avg_lean_miss1 = np.mean(
    [prob_results[b]["lean_miss_after_rewarded_rich"] for b in prob_results]
)
avg_lean_miss2 = np.mean(
    [prob_results[b]["lean_miss_after_nonrewarded_rich"] for b in prob_results]
)

fig.add_trace(
    go.Bar(
        x=["前试次富刺激有奖励", "前试次富刺激无奖励"],
        y=[avg_lean_miss1, avg_lean_miss2],
        name="Lean miss概率",
        marker_color=["royalblue", "crimson"],
        text=[f"{avg_lean_miss1:.3f}", f"{avg_lean_miss2:.3f}"],
        textposition="outside",
    ),
    row=2,
    col=1,
)

# 添加文献参考值
fig.add_trace(
    go.Scatter(
        x=["前试次富刺激有奖励", "前试次富刺激无奖励"],
        y=[0.48, 0.30],
        mode="markers",
        name="文献MDD组",
        marker=dict(size=12, color="red", symbol="diamond"),
        opacity=0.7,
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=["前试次富刺激有奖励", "前试次富刺激无奖励"],
        y=[0.49, 0.45],
        mode="markers",
        name="文献对照组",
        marker=dict(size=12, color="green", symbol="diamond"),
        opacity=0.7,
    ),
    row=2,
    col=1,
)

# 图5: Rich miss概率分析
avg_rich_miss1 = np.mean(
    [prob_results[b]["rich_miss_after_rewarded_rich"] for b in prob_results]
)
avg_rich_miss2 = np.mean(
    [prob_results[b]["rich_miss_after_rewarded_lean"] for b in prob_results]
)

fig.add_trace(
    go.Bar(
        x=["前试次富刺激有奖励", "前试次贫刺激有奖励"],
        y=[avg_rich_miss1, avg_rich_miss2],
        name="Rich miss概率",
        marker_color=["royalblue", "crimson"],
        text=[f"{avg_rich_miss1:.3f}", f"{avg_rich_miss2:.3f}"],
        textposition="outside",
    ),
    row=2,
    col=2,
)

# 添加文献参考值
fig.add_trace(
    go.Scatter(
        x=["前试次富刺激有奖励", "前试次贫刺激有奖励"],
        y=[0.12, 0.25],
        mode="markers",
        name="文献MDD组",
        marker=dict(size=12, color="red", symbol="diamond"),
        opacity=0.7,
    ),
    row=2,
    col=2,
)

fig.add_trace(
    go.Scatter(
        x=["前试次富刺激有奖励", "前试次贫刺激有奖励"],
        y=[0.13, 0.10],
        mode="markers",
        name="文献对照组",
        marker=dict(size=12, color="green", symbol="diamond"),
        opacity=0.7,
    ),
    row=2,
    col=2,
)

# 图6: 反应时对比
rt_rich_values = [rt_by_block[b]["rt_rich"] for b in blocks]
rt_lean_values = [rt_by_block[b]["rt_lean"] for b in blocks]

fig.add_trace(
    go.Scatter(
        x=blocks,
        y=rt_rich_values,
        mode="lines+markers",
        name="Rich刺激RT",
        line=dict(width=3, color="green"),
    ),
    row=2,
    col=3,
)

fig.add_trace(
    go.Scatter(
        x=blocks,
        y=rt_lean_values,
        mode="lines+markers",
        name="Lean刺激RT",
        line=dict(width=3, color="orange"),
    ),
    row=2,
    col=3,
)

# 图7: 反应时分布
fig.add_trace(
    go.Histogram(
        x=trials_df["rt"].to_numpy(),
        nbinsx=30,
        name="反应时分布",
        marker_color="purple",
        opacity=0.7,
    ),
    row=3,
    col=1,
)

# 图8: 学习曲线（以Block 0为例）
if 0 in trend_results:
    block0_trends = trend_results[0]
    x_values = [point["midpoint"] for point in block0_trends["window_accuracies"]]
    y_values = [point["accuracy"] for point in block0_trends["window_accuracies"]]

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            name="Block 0学习曲线",
            line=dict(width=2, color="blue"),
        ),
        row=3,
        col=2,
    )

# 图9: 奖励整合指数
reward_integration_idx = []
for block in blocks:
    lean_diff = (
        prob_results[block]["lean_miss_after_rewarded_rich"]
        - prob_results[block]["lean_miss_after_nonrewarded_rich"]
    )
    rich_diff = (
        prob_results[block]["rich_miss_after_rewarded_lean"]
        - prob_results[block]["rich_miss_after_rewarded_rich"]
    )
    # 综合指数：负值表示MDD模式，正值表示对照组模式
    integration_idx = (lean_diff - 0.1) - (rich_diff - 0.02)
    reward_integration_idx.append(integration_idx)

fig.add_trace(
    go.Bar(
        x=[f"Block {b}" for b in blocks],
        y=reward_integration_idx,
        name="奖励整合指数",
        marker_color=[
            "crimson" if idx < 0 else "green" for idx in reward_integration_idx
        ],
        text=[f"{idx:.3f}" for idx in reward_integration_idx],
        textposition="outside",
    ),
    row=3,
    col=3,
)

# 添加参考线
fig.add_hline(y=0, line=dict(width=1, dash="dash"), row=3, col=3)

# 更新布局
fig.update_layout(
    title=dict(
        text="PRT（概率性奖励任务）行为学分析 - 综合报告",
        font=dict(size=24, family="Arial Black"),
        x=0.5,
    ),
    height=1400,
    width=1600,
    showlegend=True,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# 更新坐标轴标签
fig.update_xaxes(title_text="Block", row=1, col=1)
fig.update_yaxes(title_text="Log b (反应偏向)", row=1, col=1)

fig.update_xaxes(title_text="Block", row=1, col=2)
fig.update_yaxes(title_text="击中率", range=[0.5, 1.0], row=1, col=2)

fig.update_xaxes(title_text="Block", row=1, col=3)
fig.update_yaxes(title_text="准确率", range=[0.5, 1.0], row=1, col=3)

fig.update_xaxes(title_text="条件", row=2, col=1)
fig.update_yaxes(title_text="Lean miss概率", range=[0, 0.6], row=2, col=1)

fig.update_xaxes(title_text="条件", row=2, col=2)
fig.update_yaxes(title_text="Rich miss概率", range=[0, 0.35], row=2, col=2)

fig.update_xaxes(title_text="Block", row=2, col=3)
fig.update_yaxes(title_text="反应时(秒)", row=2, col=3)

fig.update_xaxes(title_text="反应时(秒)", row=3, col=1)
fig.update_yaxes(title_text="频数", row=3, col=1)

fig.update_xaxes(title_text="试次位置", row=3, col=2)
fig.update_yaxes(title_text="窗口准确率", range=[0.5, 1.0], row=3, col=2)

fig.update_xaxes(title_text="Block", row=3, col=3)
fig.update_yaxes(title_text="奖励整合指数", row=3, col=3)

# 显示图表
fig.show()

# ==================== 8. 生成详细分析报告 ====================
print("\n" + "=" * 60)
print("PRT数据分析报告")
print("=" * 60)

# 计算关键指标
mean_log_b = np.mean(log_b_values)
mean_log_d = np.mean([sdt_results[b]["log_d"] for b in blocks])
lean_miss_diff = avg_lean_miss1 - avg_lean_miss2
rich_miss_diff = avg_rich_miss2 - avg_rich_miss1

print(f"\n1. 数据概况:")
print(f"   总试次数: {trials_df.height}")
print(f"   Block数量: {len(blocks)}")
print(f"   平均反应时: {mean_rt:.3f}秒")

print(f"\n2. 核心指标总结:")
print(f"   平均反应偏向(Log b): {mean_log_b:.3f}")
print(f"   平均辨别力(Log d): {mean_log_d:.3f}")
print(f"   平均Rich刺激击中率: {np.mean(rich_hit_rates):.3f}")
print(f"   平均Lean刺激击中率: {np.mean(lean_hit_rates):.3f}")
print(
    f"   击中率差异(Rich-Lean): {np.mean(rich_hit_rates) - np.mean(lean_hit_rates):.3f}"
)

print(f"\n3. 概率分析总结（关键发现）:")
print(f"   A. Lean miss概率差异: {lean_miss_diff:.3f}")
print(f"      - 文献MDD组: ~0.18 (0.48 - 0.30)")
print(f"      - 文献对照组: ~0.04 (0.49 - 0.45)")
print(f"      - 当前被试: {lean_miss_diff:.3f}")

print(f"\n   B. Rich miss概率差异: {rich_miss_diff:.3f}")
print(f"      - 文献MDD组: ~0.13 (0.25 - 0.12)")
print(f"      - 文献对照组: ~-0.03 (0.10 - 0.13)")
print(f"      - 当前被试: {rich_miss_diff:.3f}")

print(f"\n4. 临床模式评估:")
print("   =======================================")
print("   模式            | 反应偏向 | Lean miss差异 | Rich miss差异")
print("   ----------------|----------|---------------|-------------")
print(f"   文献MDD组       | <0.15    | >0.15         | >0.10")
print(f"   文献对照组      | >0.18    | <0.10         | <0.00")
print(
    f"   当前被试        | {mean_log_b:.3f}    | {lean_miss_diff:.3f}         | {rich_miss_diff:.3f}"
)
print("   =======================================")

print(f"\n5. 综合临床评估:")
if mean_log_b < 0.15 and lean_miss_diff > 0.15 and rich_miss_diff > 0.10:
    print("   🔴 强烈提示MDD模式：")
    print("      - 低反应偏向 (<0.15)")
    print("      - 无奖励后偏好迅速下降 (Lean miss差异大)")
    print("      - 对贫刺激奖励过度反应 (Rich miss差异大)")
elif mean_log_b > 0.18 and lean_miss_diff < 0.10 and rich_miss_diff < 0.00:
    print("   🟢 符合对照组模式：")
    print("      - 高反应偏向 (>0.18)")
    print("      - 良好奖励整合能力")
    print("      - 能抵抗贫刺激奖励的干扰")
else:
    print("   🟡 混合模式或中间型：")
    if mean_log_b < 0.15:
        print("      - 反应偏向较低 (可能提示快感缺乏倾向)")
    if lean_miss_diff > 0.15:
        print("      - 奖励整合能力受损 (无奖励后偏好下降明显)")
    if rich_miss_diff > 0.10:
        print("      - 对贫刺激奖励过度反应 (干扰抵抗能力弱)")

print(f"\n6. 学习能力评估:")
for block in blocks:
    acc_change = trend_results[block]["accuracy_change"]
    rt_change = trend_results[block]["rt_change"]

    print(f"\n   Block {block}:")
    print(
        f"     准确率变化: {acc_change:.3f} {'(提高)' if acc_change > 0 else '(下降或不变)'}"
    )
    print(
        f"     反应时变化: {rt_change:.3f}秒 {'(加快)' if rt_change < 0 else '(减慢或不变)'}"
    )

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)

# ==================== 9. 保存结果到文件 ====================
# 保存SDT结果
sdt_data = []
for block in blocks:
    sdt_data.append(
        {
            "block": block,
            "log_b": sdt_results[block]["log_b"],
            "log_d": sdt_results[block]["log_d"],
            "rich_hit_rate": sdt_results[block]["rich_hit_rate"],
            "lean_hit_rate": sdt_results[block]["lean_hit_rate"],
            "overall_accuracy": sdt_results[block]["overall_accuracy"],
            "hit_rate_difference": sdt_results[block]["hit_rate_difference"],
        }
    )

sdt_df = pl.DataFrame(sdt_data)

# 保存概率分析结果
prob_data = []
for block in blocks:
    prob_data.append(
        {
            "block": block,
            "lean_miss_after_rewarded_rich": prob_results[block][
                "lean_miss_after_rewarded_rich"
            ],
            "lean_miss_after_nonrewarded_rich": prob_results[block][
                "lean_miss_after_nonrewarded_rich"
            ],
            "rich_miss_after_rewarded_rich": prob_results[block][
                "rich_miss_after_rewarded_rich"
            ],
            "rich_miss_after_rewarded_lean": prob_results[block][
                "rich_miss_after_rewarded_lean"
            ],
            "rich_miss_after_nonrewarded_lean": prob_results[block].get(
                "rich_miss_after_nonrewarded_lean", 0
            ),
            "lean_miss_difference": prob_results[block]["lean_miss_after_rewarded_rich"]
            - prob_results[block]["lean_miss_after_nonrewarded_rich"],
            "rich_miss_difference": prob_results[block]["rich_miss_after_rewarded_lean"]
            - prob_results[block]["rich_miss_after_rewarded_rich"],
        }
    )

prob_df = pl.DataFrame(prob_data)

# 保存反应时结果
rt_data = []
for block in blocks:
    rt_data.append(
        {
            "block": block,
            "rt_rich": rt_by_block[block]["rt_rich"],
            "rt_lean": rt_by_block[block]["rt_lean"],
            "rt_difference": rt_by_block[block]["rt_diff"],
            "rt_correct": rt_by_block[block]["rt_correct"],
            "rt_error": rt_by_block[block]["rt_error"],
        }
    )

rt_df = pl.DataFrame(rt_data)

# 保存到CSV文件
sdt_df.write_csv("prt_sdt_results.csv")
prob_df.write_csv("prt_probability_results.csv")
rt_df.write_csv("prt_reaction_time_results.csv")

print("\n结果已保存到文件:")
print("  - prt_sdt_results.csv (SDT指标)")
print("  - prt_probability_results.csv (概率分析结果)")
print("  - prt_reaction_time_results.csv (反应时结果)")
