from pathlib import Path
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl
from plotly.subplots import make_subplots

from psycho.analysis.utils import extract_trials_by_block

# ==================== 数据处理模块 ====================


def load_and_preprocess_data(file_path: Path) -> pl.DataFrame:
    """加载并预处理数据"""
    try:
        print(f"正在读取数据文件: {file_path}")
        df = pl.read_csv(file_path)
        print(f"原始数据: {df.height} 行, {df.width} 列")

        # 提取试次数据
        trials_df = extract_trials_by_block(
            df,
            target_block_indices=[0, 1, 2],
            block_col="block_index",
            trial_col="trial_index",
        )

        if trials_df.height == 0:
            print("❌ 错误: 未找到有效的试次数据")
            return None

        print(f"成功提取 {trials_df.height} 个试次")

        # 添加分析需要的列
        trials_df = trials_df.with_columns(
            [
                (pl.col("stim") == pl.col("choice")).alias("correct"),
                pl.col("reward").gt(0).alias("rewarded"),
                (pl.col("reward") == -1).alias("error"),
            ]
        )

        return trials_df

    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
        return None


# ==================== Rich刺激识别模块 ====================


def identify_rich_stimulus(trials_df: pl.DataFrame) -> dict[int, dict[str, Any]]:
    """识别每个Block的Rich刺激"""
    print("=" * 60)
    print("识别每个Block的Rich刺激")
    print("=" * 60)

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

    return rich_stim_results


# ==================== SDT指标计算模块 ====================


def calculate_sdt_metrics(
    trials_df: pl.DataFrame, rich_stim_results: dict[int, dict[str, Any]]
) -> dict[int, dict[str, float]]:
    """计算信号检测理论指标"""
    print("\n" + "=" * 60)
    print("计算SDT指标（反应偏向和辨别力）")
    print("=" * 60)

    sdt_results = {}

    for block in sorted(trials_df["block_index"].unique()):
        block_data = trials_df.filter(pl.col("block_index") == block)
        rich_stim = rich_stim_results[block]["rich_stim"]
        lean_stim = rich_stim_results[block]["lean_stim"]

        # 提取四类试次
        rich_hit = block_data.filter(
            (pl.col("stim") == rich_stim) & (pl.col("correct"))
        ).height

        rich_miss = block_data.filter(
            (pl.col("stim") == rich_stim) & (~pl.col("correct"))
        ).height

        lean_hit = block_data.filter(
            (pl.col("stim") == lean_stim) & (pl.col("correct"))
        ).height

        lean_miss = block_data.filter(
            (pl.col("stim") == lean_stim) & (~pl.col("correct"))
        ).height

        # Hautus校正：每个单元格加0.5
        rich_hit_c = rich_hit + 0.5
        rich_miss_c = rich_miss + 0.5
        lean_hit_c = lean_hit + 0.5
        lean_miss_c = lean_miss + 0.5

        # 计算log b（反应偏向）
        if (rich_miss_c * lean_hit_c) > 0:
            log_b = 0.5 * np.log10(
                (rich_hit_c * lean_miss_c) / (rich_miss_c * lean_hit_c)
            )
        else:
            log_b = 0.0

        # 计算log d（辨别力）
        if (rich_miss_c * lean_miss_c) > 0:
            log_d = 0.5 * np.log10(
                (rich_hit_c * lean_hit_c) / (rich_miss_c * lean_miss_c)
            )
        else:
            log_d = 0.0

        # 计算击中率
        rich_total = rich_hit + rich_miss
        lean_total = lean_hit + lean_miss

        rich_hit_rate = rich_hit / rich_total if rich_total > 0 else 0
        lean_hit_rate = lean_hit / lean_total if lean_total > 0 else 0

        # 计算额外指标
        total_correct = rich_hit + lean_hit
        total_trials = rich_total + lean_total
        overall_accuracy = total_correct / total_trials if total_trials > 0 else 0

        sdt_results[block] = {
            "log_b": log_b,
            "log_d": log_d,
            "rich_hit_rate": rich_hit_rate,
            "lean_hit_rate": lean_hit_rate,
            "rich_miss_rate": 1 - rich_hit_rate,
            "lean_miss_rate": 1 - lean_hit_rate,
            "rich_hit": rich_hit,
            "rich_miss": rich_miss,
            "lean_hit": lean_hit,
            "lean_miss": lean_miss,
            "overall_accuracy": overall_accuracy,
            "hit_rate_difference": rich_hit_rate - lean_hit_rate,
        }

        print(f"Block {block}:")
        print(f"  log_b (反应偏向) = {log_b:.3f}")
        print(f"  log_d (辨别力) = {log_d:.3f}")
        print(f"  Rich刺激击中率 = {rich_hit_rate:.3f}")
        print(f"  Lean刺激击中率 = {lean_hit_rate:.3f}")
        print(f"  击中率差异(Rich-Lean) = {rich_hit_rate - lean_hit_rate:.3f}")

    return sdt_results


# ==================== 概率分析模块 ====================


def calculate_probability_analysis(
    trials_df: pl.DataFrame, rich_stim_results: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    """进行概率分析（论文图3的关键分析）"""
    print("\n" + "=" * 60)
    print("概率分析（论文图3的关键分析）")
    print("=" * 60)

    prob_results = {}

    for block in sorted(trials_df["block_index"].unique()):
        block_data = trials_df.filter(pl.col("block_index") == block).sort(
            "trial_in_block"
        )
        rich_stim = rich_stim_results[block]["rich_stim"]
        lean_stim = rich_stim_results[block]["lean_stim"]

        # 添加上一试次的信息
        block_data = block_data.with_columns(
            [
                pl.col("stim").shift(1).alias("prev_stim"),
                pl.col("rewarded").shift(1).alias("prev_rewarded"),
                pl.col("correct").shift(1).alias("prev_correct"),
            ]
        )

        # 只考虑前一试次正确的情况
        valid_data = block_data.filter(pl.col("prev_correct"))

        # 情况A: 分析lean miss概率
        lean_trials = valid_data.filter(pl.col("stim") == lean_stim)

        # A1: 前一个试次是rich且获得奖励
        cond1 = lean_trials.filter(
            (pl.col("prev_stim") == rich_stim) & (pl.col("prev_rewarded"))
        )

        # A2: 前一个试次是rich但无奖励
        cond2 = lean_trials.filter(
            (pl.col("prev_stim") == rich_stim) & (pl.col("prev_rewarded") is False)
        )

        # 计算lean miss概率
        lean_miss_prob1 = (
            (cond1.filter(pl.col("correct") is False).height / cond1.height)
            if cond1.height > 0
            else 0
        )
        lean_miss_prob2 = (
            (cond2.filter(pl.col("correct") is False).height / cond2.height)
            if cond2.height > 0
            else 0
        )

        # 情况B: 分析rich miss概率
        rich_trials = valid_data.filter(pl.col("stim") == rich_stim)

        # B1: 前一个试次是rich且获得奖励
        cond3 = rich_trials.filter(
            (pl.col("prev_stim") == rich_stim) & (pl.col("prev_rewarded"))
        )

        # B2: 前一个试次是lean且获得奖励
        cond4 = rich_trials.filter(
            (pl.col("prev_stim") == lean_stim) & (pl.col("prev_rewarded"))
        )

        # 计算rich miss概率
        rich_miss_prob1 = (
            (cond3.filter(pl.col("correct") is False).height / cond3.height)
            if cond3.height > 0
            else 0
        )
        rich_miss_prob2 = (
            (cond4.filter(pl.col("correct") is False).height / cond4.height)
            if cond4.height > 0
            else 0
        )

        prob_results[block] = {
            "lean_miss_after_rewarded_rich": lean_miss_prob1,
            "lean_miss_after_nonrewarded_rich": lean_miss_prob2,
            "rich_miss_after_rewarded_rich": rich_miss_prob1,
            "rich_miss_after_rewarded_lean": rich_miss_prob2,
            "counts": {
                "cond1": cond1.height,
                "cond2": cond2.height,
                "cond3": cond3.height,
                "cond4": cond4.height,
            },
        }

        print(f"\nBlock {block}:")
        print(
            f"  1. Lean miss概率（前试次富刺激有奖励）: {lean_miss_prob1:.3f} (n={cond1.height})"
        )
        print(
            f"  2. Lean miss概率（前试次富刺激无奖励）: {lean_miss_prob2:.3f} (n={cond2.height})"
        )
        print(f"  差异（1-2）: {lean_miss_prob1 - lean_miss_prob2:.3f}")
        print(
            f"  3. Rich miss概率（前试次富刺激有奖励）: {rich_miss_prob1:.3f} (n={cond3.height})"
        )
        print(
            f"  4. Rich miss概率（前试次贫刺激有奖励）: {rich_miss_prob2:.3f} (n={cond4.height})"
        )
        print(f"  差异（4-3）: {rich_miss_prob2 - rich_miss_prob1:.3f}")

    return prob_results


# ==================== 反应时分析模块 ====================


def analyze_reaction_time(
    trials_df: pl.DataFrame, rich_stim_results: dict[int, dict[str, Any]]
) -> dict[int, dict[str, float]]:
    """分析反应时"""
    print("\n" + "=" * 60)
    print("反应时分析")
    print("=" * 60)

    rt_by_block = {}

    # 总体反应时统计
    mean_rt = trials_df["rt"].mean()
    median_rt = trials_df["rt"].median()
    std_rt = trials_df["rt"].std()

    print("总体反应时:")
    print(f"  均值: {mean_rt:.3f}秒")
    print(f"  中位数: {median_rt:.3f}秒")
    print(f"  标准差: {std_rt:.3f}秒")

    # 按Block分析
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

    return rt_by_block


# ==================== 性能趋势分析模块 ====================


def analyze_performance_trends(trials_df: pl.DataFrame) -> dict[int, dict[str, Any]]:
    """分析性能随时间和试次的变化趋势"""
    print("\n" + "=" * 60)
    print("性能趋势分析")
    print("=" * 60)

    results = {}

    for block in sorted(trials_df["block_index"].unique()):
        block_data = trials_df.filter(pl.col("block_index") == block).sort(
            "trial_in_block"
        )

        # 计算学习曲线：前1/3 vs 后1/3试次
        total_trials = block_data.height
        third = total_trials // 3

        if third > 0:
            early_trials = block_data.slice(0, third)
            late_trials = block_data.slice(total_trials - third, third)

            early_accuracy = early_trials.filter(pl.col("correct")).height / third
            late_accuracy = late_trials.filter(pl.col("correct")).height / third

            # 反应时变化
            early_rt = early_trials["rt"].mean()
            late_rt = late_trials["rt"].mean()

            results[block] = {
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


# ==================== 可视化模块 ====================


def create_visualizations(
    sdt_results: dict[int, dict[str, float]],
    prob_results: dict[int, dict[str, Any]],
    rt_by_block: dict[int, dict[str, float]],
    trend_results: dict[int, dict[str, Any]],
    result_dir: Path,
) -> go.Figure:
    """创建可视化图表"""
    print("\n" + "=" * 60)
    print("创建可视化图表")
    print("=" * 60)

    blocks = sorted(sdt_results.keys())

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
    log_b_values = [sdt_results[b]["log_b"] for b in blocks]

    # 文献参考值
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

    # 图8: 学习曲线（以Block 0为例）
    if 0 in trend_results:
        block0_early = trend_results[0]["early_accuracy"]
        block0_late = trend_results[0]["late_accuracy"]

        fig.add_trace(
            go.Scatter(
                x=["早期", "晚期"],
                y=[block0_early, block0_late],
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

    fig.update_xaxes(title_text="学习阶段", row=3, col=2)
    fig.update_yaxes(title_text="准确率", range=[0.5, 1.0], row=3, col=2)

    fig.update_xaxes(title_text="Block", row=3, col=3)
    fig.update_yaxes(title_text="奖励整合指数", row=3, col=3)

    # 保存图表
    html_path = result_dir / "prt_visualization.html"
    fig.write_html(str(html_path))
    print(f"可视化图表已保存: {html_path}")

    return fig


# ==================== 报告生成模块 ====================


def generate_report(
    trials_df: pl.DataFrame,
    sdt_results: dict[int, dict[str, float]],
    prob_results: dict[int, dict[str, Any]],
    rt_by_block: dict[int, dict[str, float]],
    trend_results: dict[int, dict[str, Any]],
    result_dir: Path,
) -> dict[str, Any]:
    """生成分析报告并保存结果"""
    print("\n" + "=" * 60)
    print("PRT数据分析报告")
    print("=" * 60)

    # 计算关键指标
    blocks = sorted(sdt_results.keys())
    mean_log_b = np.mean([sdt_results[b]["log_b"] for b in blocks])
    mean_log_d = np.mean([sdt_results[b]["log_d"] for b in blocks])
    rich_hit_rates = [sdt_results[b]["rich_hit_rate"] for b in blocks]
    lean_hit_rates = [sdt_results[b]["lean_hit_rate"] for b in blocks]

    avg_lean_miss1 = np.mean(
        [prob_results[b]["lean_miss_after_rewarded_rich"] for b in prob_results]
    )
    avg_lean_miss2 = np.mean(
        [prob_results[b]["lean_miss_after_nonrewarded_rich"] for b in prob_results]
    )
    avg_rich_miss1 = np.mean(
        [prob_results[b]["rich_miss_after_rewarded_rich"] for b in prob_results]
    )
    avg_rich_miss2 = np.mean(
        [prob_results[b]["rich_miss_after_rewarded_lean"] for b in prob_results]
    )

    lean_miss_diff = avg_lean_miss1 - avg_lean_miss2
    rich_miss_diff = avg_rich_miss2 - avg_rich_miss1

    # 打印报告
    print("\n1. 数据概况:")
    print(f"   总试次数: {trials_df.height}")
    print(f"   Block数量: {len(blocks)}")
    print(f"   平均反应时: {trials_df['rt'].mean():.3f}秒")

    print("\n2. 核心指标总结:")
    print(f"   平均反应偏向(Log b): {mean_log_b:.3f}")
    print(f"   平均辨别力(Log d): {mean_log_d:.3f}")
    print(f"   平均Rich刺激击中率: {np.mean(rich_hit_rates):.3f}")
    print(f"   平均Lean刺激击中率: {np.mean(lean_hit_rates):.3f}")
    print(
        f"   击中率差异(Rich-Lean): {np.mean(rich_hit_rates) - np.mean(lean_hit_rates):.3f}"
    )

    print("\n3. 概率分析总结（关键发现）:")
    print(f"   A. Lean miss概率差异: {lean_miss_diff:.3f}")
    print("      - 文献MDD组: ~0.18 (0.48 - 0.30)")
    print("      - 文献对照组: ~0.04 (0.49 - 0.45)")
    print(f"      - 当前被试: {lean_miss_diff:.3f}")

    print(f"\n   B. Rich miss概率差异: {rich_miss_diff:.3f}")
    print("      - 文献MDD组: ~0.13 (0.25 - 0.12)")
    print("      - 文献对照组: ~-0.03 (0.10 - 0.13)")
    print(f"      - 当前被试: {rich_miss_diff:.3f}")

    print("\n4. 临床模式评估:")
    print("   =======================================")
    print("   模式            | 反应偏向 | Lean miss差异 | Rich miss差异")
    print("   ----------------|----------|---------------|-------------")
    print("   文献MDD组       | <0.15    | >0.15         | >0.10")
    print("   文献对照组      | >0.18    | <0.10         | <0.00")
    print(
        f"   当前被试        | {mean_log_b:.3f}    | {lean_miss_diff:.3f}         | {rich_miss_diff:.3f}"
    )
    print("   =======================================")

    print("\n5. 综合临床评估:")
    if mean_log_b < 0.15 and lean_miss_diff > 0.15 and rich_miss_diff > 0.10:
        assessment = "MDD模式"
        print("   🔴 强烈提示MDD模式：")
        print("      - 低反应偏向 (<0.15)")
        print("      - 无奖励后偏好迅速下降 (Lean miss差异大)")
        print("      - 对贫刺激奖励过度反应 (Rich miss差异大)")
    elif mean_log_b > 0.18 and lean_miss_diff < 0.10 and rich_miss_diff < 0.00:
        assessment = "对照组模式"
        print("   🟢 符合对照组模式：")
        print("      - 高反应偏向 (>0.18)")
        print("      - 良好奖励整合能力")
        print("      - 能抵抗贫刺激奖励的干扰")
    else:
        assessment = "混合模式"
        print("   🟡 混合模式或中间型：")
        if mean_log_b < 0.15:
            print("      - 反应偏向较低 (可能提示快感缺乏倾向)")
        if lean_miss_diff > 0.15:
            print("      - 奖励整合能力受损 (无奖励后偏好下降明显)")
        if rich_miss_diff > 0.10:
            print("      - 对贫刺激奖励过度反应 (干扰抵抗能力弱)")

    # 保存结果到文件
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
    sdt_df.write_csv(result_dir / "prt_sdt_results.csv")

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
                "lean_miss_difference": prob_results[block][
                    "lean_miss_after_rewarded_rich"
                ]
                - prob_results[block]["lean_miss_after_nonrewarded_rich"],
                "rich_miss_difference": prob_results[block][
                    "rich_miss_after_rewarded_lean"
                ]
                - prob_results[block]["rich_miss_after_rewarded_rich"],
            }
        )

    prob_df = pl.DataFrame(prob_data)
    prob_df.write_csv(result_dir / "prt_probability_results.csv")

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
    rt_df.write_csv(result_dir / "prt_reaction_time_results.csv")

    print(f"\n结果已保存到: {result_dir}")
    print("  - prt_sdt_results.csv (SDT指标)")
    print("  - prt_probability_results.csv (概率分析结果)")
    print("  - prt_reaction_time_results.csv (反应时结果)")
    print("  - prt_visualization.html (可视化图表)")

    # 返回汇总结果
    return {
        "data_summary": {
            "total_trials": trials_df.height,
            "num_blocks": len(blocks),
            "mean_rt": float(trials_df["rt"].mean()),
        },
        "sdt_metrics": {
            "mean_log_b": float(mean_log_b),
            "mean_log_d": float(mean_log_d),
            "mean_rich_hit_rate": float(np.mean(rich_hit_rates)),
            "mean_lean_hit_rate": float(np.mean(lean_hit_rates)),
        },
        "probability_analysis": {
            "lean_miss_difference": float(lean_miss_diff),
            "rich_miss_difference": float(rich_miss_diff),
        },
        "clinical_assessment": assessment,
    }


# ==================== 主分析函数 ====================


def analyze_prt_data(
    df: pl.DataFrame,
    target_blocks: list[int] = [0, 1, 2],
    result_dir: Path = Path("results"),
) -> dict[str, Any]:
    """
    主分析函数：执行PRT数据分析

    参数:
    ----------
    df : pl.DataFrame
        原始数据
    target_blocks : list[int]
        目标区块列表
    result_dir : Path
        结果保存目录

    返回:
    -------
    dict[str, Any]
        分析结果汇总
    """
    print("开始PRT数据分析...")

    # 1. 加载并预处理数据
    trials_df = extract_trials_by_block(
        df,
        target_block_indices=target_blocks,
        block_col="block_index",
        trial_col="trial_index",
    )

    if trials_df.height == 0:
        print("❌ 错误: 未找到有效的试次数据")
        return {}

    # 添加分析需要的列
    trials_df = trials_df.with_columns(
        [
            (pl.col("stim") == pl.col("choice")).alias("correct"),
            pl.col("reward").gt(0).alias("rewarded"),
            (pl.col("reward") == -1).alias("error"),
        ]
    )

    # 2. 识别Rich刺激
    rich_stim_results = identify_rich_stimulus(trials_df)

    # 3. 计算SDT指标
    sdt_results = calculate_sdt_metrics(trials_df, rich_stim_results)

    # 4. 概率分析
    prob_results = calculate_probability_analysis(trials_df, rich_stim_results)

    # 5. 反应时分析
    rt_by_block = analyze_reaction_time(trials_df, rich_stim_results)

    # 6. 性能趋势分析
    trend_results = analyze_performance_trends(trials_df)

    # 7. 创建可视化
    _ = create_visualizations(
        sdt_results, prob_results, rt_by_block, trend_results, result_dir
    )

    # 8. 生成报告
    results = generate_report(
        trials_df, sdt_results, prob_results, rt_by_block, trend_results, result_dir
    )

    return results


def run_prt_analysis(cfg=None):
    """运行PRT（概率性奖励任务）分析"""
    print("=" * 60)
    print("PRT（概率性奖励任务）分析系统")
    print("=" * 60)

    # 获取文件路径
    file_input = input("请输入数据文件路径: \n").strip("'").strip()

    file_path = Path(file_input.strip("'").strip('"')).resolve()

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    # 读取数据
    print(f"正在读取数据文件: {file_path}")
    df = pl.read_csv(file_path)

    # 设置结果目录
    if cfg is None:
        result_dir = file_path.parent / "prt_results"
        result_dir = file_path.parent.parent / "results" / "prt_analysis"
    else:
        result_dir = Path(cfg.result_dir)

    result_dir.mkdir(parents=True, exist_ok=True)

    # 运行分析
    results = analyze_prt_data(df=df, target_blocks=[0, 1, 2], result_dir=result_dir)

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_prt_analysis()
