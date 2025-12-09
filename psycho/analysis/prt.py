from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl
from omegaconf import DictConfig
from plotly.subplots import make_subplots

from psycho.analysis.utils import DataUtils, extract_trials_by_block

REFERENCE_VALUES = {
    "control": {
        "log_b": [0.19, 0.24, 0.23],  # 对照组log b (三个block)
        "rich_hit_rate": 0.88,
        "lean_hit_rate": 0.75,
        "lean_miss_after_rewarded_rich": 0.26,
        "lean_miss_after_nonrewarded_rich": 0.23,
        "rich_miss_after_rewarded_rich": 0.13,
        "rich_miss_after_rewarded_lean": 0.11,
    },
    "mdd": {
        "log_b": [0.08, 0.12, 0.10],  # MDD组log b (三个block)
        "rich_hit_rate": 0.86,
        "lean_hit_rate": 0.77,
        "lean_miss_after_rewarded_rich": 0.25,
        "lean_miss_after_nonrewarded_rich": 0.15,
        "rich_miss_after_rewarded_rich": 0.11,
        "rich_miss_after_rewarded_lean": 0.16,
    },
}

all_raw_df = None


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


def identify_rich_stimulus(trials_df: pl.DataFrame) -> dict[int, dict[str, Any]]:
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


def calculate_sdt_metrics(
    trials_df: pl.DataFrame, rich_stim_results: dict[int, dict[str, Any]]
) -> dict[int, dict[str, float]]:
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
        print(f"    文献对照组参考: {REFERENCE_VALUES['control']['log_b'][block]:.3f}")
        print(f"    文献MDD组参考: {REFERENCE_VALUES['mdd']['log_b'][block]:.3f}")
        print(f"  log_d (辨别力) = {log_d:.3f}")
        print(f"  Rich刺激击中率 = {rich_hit_rate:.3f}")
        print(f"  Lean刺激击中率 = {lean_hit_rate:.3f}")
        print(f"  击中率差异(Rich-Lean) = {rich_hit_rate - lean_hit_rate:.3f}")

    return sdt_results


def calculate_probability_analysis_improved(
    trials_df: pl.DataFrame, rich_stim_results: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    print("\n" + "=" * 60)
    print("概率分析")
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
            (pl.col("prev_stim") == rich_stim) & (~pl.col("prev_rewarded"))
        )

        # 计算lean miss概率
        lean_miss_rate1 = (
            (cond1.filter(~pl.col("correct")).height / cond1.height)
            if cond1.height > 0
            else 0
        )
        lean_miss_rate2 = (
            (cond2.filter(~pl.col("correct")).height / cond2.height)
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
        rich_miss_rate1 = (
            (cond3.filter(~pl.col("correct")).height / cond3.height)
            if cond3.height > 0
            else 0
        )
        rich_miss_rate2 = (
            (cond4.filter(~pl.col("correct")).height / cond4.height)
            if cond4.height > 0
            else 0
        )

        prob_results[block] = {
            "lean_miss_after_rewarded_rich": lean_miss_rate1,
            "lean_miss_after_nonrewarded_rich": lean_miss_rate2,
            "rich_miss_after_rewarded_rich": rich_miss_rate1,
            "rich_miss_after_rewarded_lean": rich_miss_rate2,
            "counts": {
                "cond1": cond1.height,
                "cond2": cond2.height,
                "cond3": cond3.height,
                "cond4": cond4.height,
            },
        }

        print(f"\nBlock {block}:")
        print(
            f"  1. Lean miss概率（前试次富刺激有奖励）: {lean_miss_rate1:.3f} (n={cond1.height})"
        )
        print(
            f"     文献对照组参考: {REFERENCE_VALUES['control']['lean_miss_after_rewarded_rich']:.3f}"
        )
        print(
            f"     文献MDD组参考: {REFERENCE_VALUES['mdd']['lean_miss_after_rewarded_rich']:.3f}"
        )

        print(
            f"  2. Lean miss概率（前试次富刺激无奖励）: {lean_miss_rate2:.3f} (n={cond2.height})"
        )
        print(
            f"     文献对照组参考: {REFERENCE_VALUES['control']['lean_miss_after_nonrewarded_rich']:.3f}"
        )
        print(
            f"     文献MDD组参考: {REFERENCE_VALUES['mdd']['lean_miss_after_nonrewarded_rich']:.3f}"
        )

        print(
            f"  3. Rich miss概率（前试次富刺激有奖励）: {rich_miss_rate1:.3f} (n={cond3.height})"
        )
        print(
            f"     文献对照组参考: {REFERENCE_VALUES['control']['rich_miss_after_rewarded_rich']:.3f}"
        )
        print(
            f"     文献MDD组参考: {REFERENCE_VALUES['mdd']['rich_miss_after_rewarded_rich']:.3f}"
        )

        print(
            f"  4. Rich miss概率（前试次贫刺激有奖励）: {rich_miss_rate2:.3f} (n={cond4.height})"
        )
        print(
            f"     文献对照组参考: {REFERENCE_VALUES['control']['rich_miss_after_rewarded_lean']:.3f}"
        )
        print(
            f"     文献MDD组参考: {REFERENCE_VALUES['mdd']['rich_miss_after_rewarded_lean']:.3f}"
        )

    return prob_results


def analyze_reaction_time(
    trials_df: pl.DataFrame, rich_stim_results: dict[int, dict[str, Any]]
) -> dict[int, dict[str, float]]:
    print("\n" + "=" * 60)
    print("反应时分析")
    print("=" * 60)

    rt_by_block = {}

    # 总体反应时统计
    mean_rt = trials_df["rt_clean"].mean()
    median_rt = trials_df["rt_clean"].median()
    std_rt = trials_df["rt_clean"].std()

    print("总体反应时:")
    print(f"  均值: {mean_rt:.3f}秒")
    print(f"  中位数: {median_rt:.3f}秒")
    print(f"  标准差: {std_rt:.3f}秒")

    # 按Block分析
    for block in sorted(trials_df["block_index"].unique()):
        block_data = trials_df.filter(pl.col("block_index") == block)
        rich_stim = rich_stim_results[block]["rich_stim"]

        # Rich刺激的反应时
        rt_rich = block_data.filter(pl.col("stim") == rich_stim)["rt_clean"].mean()

        # Lean刺激的反应时
        rt_lean = block_data.filter(pl.col("stim") != rich_stim)["rt_clean"].mean()

        # 正确和错误试次的反应时
        rt_correct = block_data.filter(pl.col("correct"))["rt_clean"].mean()
        rt_error = block_data.filter(~pl.col("correct"))["rt_clean"].mean()

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


def analyze_performance_trends(trials_df: pl.DataFrame) -> dict[int, dict[str, Any]]:
    print("\n" + "=" * 60)
    print("表现趋势分析")
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
            early_rt = early_trials["rt_clean"].mean()
            late_rt = late_trials["rt_clean"].mean()

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


def calculate_diagnostic_indices(
    sdt_results: dict[int, dict[str, float]],
    prob_results: dict[int, dict[str, Any]],
    rt_by_block: dict[int, dict[str, float]],
) -> dict[str, float]:
    print("\n" + "=" * 60)
    print("计算指标")
    print("=" * 60)

    blocks = sorted(sdt_results.keys())

    # 平均log b
    mean_log_b = np.mean([sdt_results[b]["log_b"] for b in blocks])

    # 平均Rich击中率
    mean_rich_hit_rate = np.mean([sdt_results[b]["rich_hit_rate"] for b in blocks])

    # 平均Lean击中率
    mean_lean_hit_rate = np.mean([sdt_results[b]["lean_hit_rate"] for b in blocks])

    # 反应时差异
    mean_rt_diff = np.mean([rt_by_block[b]["rt_diff"] for b in blocks])

    diagnostic_indices = {
        "mean_log_b": mean_log_b,
        "mean_rich_hit_rate": mean_rich_hit_rate,
        "mean_lean_hit_rate": mean_lean_hit_rate,
        "mean_rt_diff": mean_rt_diff,
    }

    print("关键指标:")
    print(f"  平均log b: {mean_log_b:.3f}")

    return diagnostic_indices


def generate_interpretation(
    diagnostic_indices: dict[str, float],
    sdt_results: dict[int, dict[str, float]],
    prob_results: dict[int, dict[str, Any]],
) -> str:
    blocks = sorted(sdt_results.keys())

    interpretation = []
    interpretation.append("=" * 60)
    interpretation.append("PRT解释报告")
    interpretation.append("=" * 60)
    interpretation.append("")

    interpretation.append("1. 总体概况:")
    mean_accuracy = np.mean([sdt_results[b]["overall_accuracy"] for b in blocks])
    interpretation.append(f"   - 总体准确率: {mean_accuracy:.1%}")
    interpretation.append(
        f"   - 平均反应偏向(log b): {diagnostic_indices['mean_log_b']:.3f}"
    )
    interpretation.append("")

    interpretation.append("2. 反应偏向分析:")
    mean_log_b = diagnostic_indices["mean_log_b"]
    control_ref = np.mean(REFERENCE_VALUES["control"]["log_b"])
    mdd_ref = np.mean(REFERENCE_VALUES["mdd"]["log_b"])

    if mean_log_b > (control_ref + mdd_ref) / 2:
        interpretation.append(
            f"   - 反应偏向({mean_log_b:.3f})相对较高，接近对照组水平({control_ref:.3f})"
        )
        interpretation.append("   - 提示对奖励的反应相对正常")
    elif mean_log_b < (control_ref + mdd_ref) / 2:
        interpretation.append(
            f"   - 反应偏向({mean_log_b:.3f})相对较低，接近MDD组水平({mdd_ref:.3f})"
        )
        interpretation.append("   - 提示可能存在奖励反应性降低")
    else:
        interpretation.append(f"   - 反应偏向({mean_log_b:.3f})处于中间水平")

    interpretation.append("")

    return "\n".join(interpretation)


def create_visualizations(
    sdt_results: dict[int, dict[str, float]],
    prob_results: dict[int, dict[str, Any]],
    rt_by_block: dict[int, dict[str, float]],
    trend_results: dict[int, dict[str, Any]],
    diagnostic_indices: dict[str, float],
    result_dir: Path,
) -> tuple[go.Figure, go.Figure]:
    print("\n" + "=" * 60)
    print("创建可视化图表")
    print("=" * 60)

    blocks = sorted(sdt_results.keys())

    fig_main = make_subplots(
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
            None,
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "scatter"}, None],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.15,
    )

    # 图1: 反应偏向(Log b)随Block变化
    log_b_values = [sdt_results[b]["log_b"] for b in blocks]

    fig_main.add_trace(
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

    fig_main.add_trace(
        go.Scatter(
            x=blocks,
            y=REFERENCE_VALUES["mdd"]["log_b"],
            mode="lines",
            name="文献MDD组",
            line=dict(width=2, color="red", dash="dash"),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    fig_main.add_trace(
        go.Scatter(
            x=blocks,
            y=REFERENCE_VALUES["control"]["log_b"],
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

    fig_main.add_trace(
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

    fig_main.add_trace(
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

    # 添加参考线
    fig_main.add_hline(
        y=REFERENCE_VALUES["control"]["rich_hit_rate"],
        annotation_text="HC Rich",
        annotation_position="right",
        line=dict(width=1, dash="dash", color="green"),
        row=1,
        col=2,
    )
    fig_main.add_hline(
        y=REFERENCE_VALUES["control"]["lean_hit_rate"],
        annotation_text="HC Lean",
        annotation_position="right",
        line=dict(width=1, dash="dash", color="darkgreen"),
        row=1,
        col=2,
    )
    fig_main.add_hline(
        y=REFERENCE_VALUES["mdd"]["rich_hit_rate"],
        annotation_text="MDD Rich",
        annotation_position="right",
        line=dict(width=1, dash="dash", color="red"),
        row=1,
        col=2,
    )
    fig_main.add_hline(
        y=REFERENCE_VALUES["mdd"]["lean_hit_rate"],
        annotation_text="MDD Lean",
        annotation_position="right",
        line=dict(width=1, dash="dash", color="darkred"),
        row=1,
        col=2,
    )

    fig_main.update_xaxes(
        ticktext=[f"Block {b}" for b in blocks], tickvals=x_positions, row=1, col=2
    )

    # 图3: 准确率趋势
    fig_main.add_trace(
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

    fig_main.add_trace(
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
    fig_main.add_trace(
        go.Scatter(
            x=["前试次富刺激有奖励", "前试次富刺激无奖励"],
            y=[
                REFERENCE_VALUES["mdd"]["lean_miss_after_rewarded_rich"],
                REFERENCE_VALUES["mdd"]["lean_miss_after_nonrewarded_rich"],
            ],
            mode="markers",
            name="文献MDD组",
            marker=dict(size=12, color="red", symbol="diamond"),
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    fig_main.add_trace(
        go.Scatter(
            x=["前试次富刺激有奖励", "前试次富刺激无奖励"],
            y=[
                REFERENCE_VALUES["control"]["lean_miss_after_rewarded_rich"],
                REFERENCE_VALUES["control"]["lean_miss_after_nonrewarded_rich"],
            ],
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

    fig_main.add_trace(
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
    fig_main.add_trace(
        go.Scatter(
            x=["前试次富刺激有奖励", "前试次贫刺激有奖励"],
            y=[
                REFERENCE_VALUES["mdd"]["rich_miss_after_rewarded_rich"],
                REFERENCE_VALUES["mdd"]["rich_miss_after_rewarded_lean"],
            ],
            mode="markers",
            name="文献MDD组",
            marker=dict(size=12, color="red", symbol="diamond"),
            opacity=0.7,
        ),
        row=2,
        col=2,
    )

    fig_main.add_trace(
        go.Scatter(
            x=["前试次富刺激有奖励", "前试次贫刺激有奖励"],
            y=[
                REFERENCE_VALUES["control"]["rich_miss_after_rewarded_rich"],
                REFERENCE_VALUES["control"]["rich_miss_after_rewarded_lean"],
            ],
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

    fig_main.add_trace(
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

    fig_main.add_trace(
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
    fig_main.add_trace(
        go.Histogram(
            x=all_raw_df["rt"].drop_nulls().to_numpy(),
            name="反应时分布",
            nbinsx=50,
            marker_color="skyblue",
            opacity=0.7,
        ),
        row=3,
        col=1,
    )

    # 图8: 学习曲线（以Block 0为例）
    if 0 in trend_results:
        block0_early = trend_results[0]["early_accuracy"]
        block0_late = trend_results[0]["late_accuracy"]

        fig_main.add_trace(
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

    # 更新坐标轴标签
    fig_main.update_xaxes(title_text="Block", row=1, col=1)
    fig_main.update_yaxes(title_text="Log b (反应偏向)", row=1, col=1)

    fig_main.update_xaxes(title_text="Block", row=1, col=2)
    fig_main.update_yaxes(title_text="击中率", range=[0.5, 1.0], row=1, col=2)

    fig_main.update_xaxes(title_text="Block", row=1, col=3)
    fig_main.update_yaxes(title_text="准确率", range=[0.5, 1.0], row=1, col=3)

    fig_main.update_xaxes(title_text="条件", row=2, col=1)
    fig_main.update_yaxes(title_text="Lean miss概率", range=[0, 0.6], row=2, col=1)

    fig_main.update_xaxes(title_text="条件", row=2, col=2)
    fig_main.update_yaxes(title_text="Rich miss概率", range=[0, 0.35], row=2, col=2)

    fig_main.update_xaxes(title_text="Block", row=2, col=3)
    fig_main.update_yaxes(title_text="反应时(秒)", row=2, col=3)

    fig_main.update_xaxes(title_text="学习阶段", row=3, col=2)
    fig_main.update_yaxes(title_text="准确率", range=[0.5, 1.0], row=3, col=2)

    # 保存图表
    html_path_main = result_dir / "prt_visualization.html"
    fig_main.write_html(str(html_path_main))
    print(f"主分析图表已保存: {html_path_main}")

    return fig_main


def generate_report(
    trials_df: pl.DataFrame,
    sdt_results: dict[int, dict[str, float]],
    prob_results: dict[int, dict[str, Any]],
    rt_by_block: dict[int, dict[str, float]],
    trend_results: dict[int, dict[str, Any]],
    diagnostic_indices: dict[str, float],
    clinical_interpretation: str,
    result_dir: Path,
) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("PRT数据分析报告")
    print("=" * 60)

    # 计算关键指标
    blocks = sorted(sdt_results.keys())
    mean_log_b = np.mean([sdt_results[b]["log_b"] for b in blocks])
    mean_log_d = np.mean([sdt_results[b]["log_d"] for b in blocks])
    rich_hit_rates = [sdt_results[b]["rich_hit_rate"] for b in blocks]
    lean_hit_rates = [sdt_results[b]["lean_hit_rate"] for b in blocks]

    # 计算总体准确率和反应时
    overall_accuracy = trials_df.filter(pl.col("correct")).height / trials_df.height
    mean_rt = trials_df["rt"].mean()

    # 打印报告
    print("\n1. 数据概况:")
    print(f"   总试次数: {trials_df.height}")
    print(f"   Block数量: {len(blocks)}")
    print(f"   总体准确率: {overall_accuracy:.3f}")
    print(f"   平均反应时: {mean_rt:.3f}秒")

    print("\n2. 核心指标总结:")
    print(f"   平均反应偏向(Log b): {mean_log_b:.3f}")
    print(f"   平均辨别力(Log d): {mean_log_d:.3f}")
    print(f"   平均Rich刺激击中率: {np.mean(rich_hit_rates):.3f}")
    print(f"   平均Lean刺激击中率: {np.mean(lean_hit_rates):.3f}")
    print(
        f"   击中率差异(Rich-Lean): {np.mean(rich_hit_rates) - np.mean(lean_hit_rates):.3f}"
    )

    print("\n3. 概率分析总结（关键发现）:")
    print(f"   A. Response Bias差异: {mean_log_b:.3f}")
    print("      - 文献对照组: ~0.20, ~0.23, ~0.22")
    print("      - 文献MDD组: ~0.08, ~0.11, ~0.10")
    print(f"      - 当前被试: {mean_log_b:.3f}")

    print(
        f"   B. 击中率差异(Hit Rate): Rich[{np.mean(rich_hit_rates):.3f}], Lean[{np.mean(lean_hit_rates):.3f}]"
    )
    print("      - 文献对照组: Rich[0.88±0.06], Lean[0.75±0.03]")
    print("      - 文献MDD组: Rich[0.86±0.08], Lean[0.77±0.05]")
    print(
        f"      - 当前被试: Rich[{np.mean(rich_hit_rates):.3f}], Lean[{np.mean(lean_hit_rates):.3f}]"
    )

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

    # 保存诊断指标
    diagnostic_df = pl.DataFrame([diagnostic_indices])
    diagnostic_df.write_csv(result_dir / "prt_diagnostic_indices.csv")

    # 保存综合报告
    with open(result_dir / "prt_summary_report.txt", "w", encoding="utf-8") as f:
        f.write("PRT（概率性奖励任务）数据分析报告\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. 数据概况:\n")
        f.write(f"   总试次数: {trials_df.height}\n")
        f.write(f"   Block数量: {len(blocks)}\n")
        f.write(f"   总体准确率: {overall_accuracy:.3f}\n")
        f.write(f"   平均反应时: {mean_rt:.3f}秒\n\n")

        f.write("2. 核心指标总结:\n")
        f.write(f"   平均反应偏向(Log b): {mean_log_b:.3f}\n")
        f.write(f"   平均辨别力(Log d): {mean_log_d:.3f}\n")
        f.write(f"   平均Rich刺激击中率: {np.mean(rich_hit_rates):.3f}\n")
        f.write(f"   平均Lean刺激击中率: {np.mean(lean_hit_rates):.3f}\n\n")

    print(f"\n结果已保存到: {result_dir}")
    print("  - prt_sdt_results.csv (SDT指标)")
    print("  - prt_probability_results.csv (概率分析结果)")
    print("  - prt_reaction_time_results.csv (反应时结果)")
    print("  - prt_diagnostic_indices.csv (诊断指标)")
    print("  - prt_visualization.html (主分析图表)")
    print("  - prt_summary_report.txt (综合报告)")

    # 返回汇总结果
    return {
        "data_summary": {
            "total_trials": trials_df.height,
            "num_blocks": len(blocks),
            "overall_accuracy": float(overall_accuracy),
            "mean_rt": float(mean_rt),
        },
        "sdt_metrics": {
            "mean_log_b": float(mean_log_b),
            "mean_log_d": float(mean_log_d),
            "mean_rich_hit_rate": float(np.mean(rich_hit_rates)),
            "mean_lean_hit_rate": float(np.mean(lean_hit_rates)),
        },
        "diagnostic_indices": diagnostic_indices,
    }


def analyze_prt_data(
    df: pl.DataFrame,
    target_blocks: list[int] = [0, 1, 2],
    result_dir: Path = Path("results"),
) -> dict[str, Any]:
    global all_raw_df
    print("开始PRT数据分析...")

    # 记录开始时间
    start_time = datetime.now()

    # 1. 提取试次数据
    trials_df = extract_trials_by_block(
        df,
        target_block_indices=target_blocks,
        block_col="block_index",
        trial_col="trial_index",
    )

    if trials_df.height == 0:
        print("❌ 错误: 未找到有效的试次数据")
        return {}

    all_raw_df = trials_df.clone()
    # 添加分析需要的列
    # [ ] 无反应的列是否需要额外处理
    trials_df = trials_df.with_columns(
        [
            pl.when(pl.col("rt") < 0.1)
            .then(0.1)
            .when(pl.col("rt").is_null())
            .then(pl.col("rt").max() + 0.6)
            .otherwise(pl.col("rt"))
            .alias("rt_clean"),
            (pl.col("stim") == pl.col("choice")).alias("correct"),
            pl.col("reward").gt(0).alias("rewarded"),
            (pl.col("reward") == -1).alias("error"),
        ]
    )

    # 识别Rich刺激
    rich_stim_results = identify_rich_stimulus(trials_df)

    # 计算SDT指标
    sdt_results = calculate_sdt_metrics(trials_df, rich_stim_results)

    # 概率分析
    prob_results = calculate_probability_analysis_improved(trials_df, rich_stim_results)

    # 反应时分析
    rt_by_block = analyze_reaction_time(trials_df, rich_stim_results)

    # 性能趋势分析
    trend_results = analyze_performance_trends(trials_df)

    # 计算指标
    diagnostic_indices = calculate_diagnostic_indices(
        sdt_results, prob_results, rt_by_block
    )

    # 8. 生成解释
    clinical_interpretation = generate_interpretation(
        diagnostic_indices, sdt_results, prob_results
    )

    # 9. 创建可视化
    fig_main = create_visualizations(  # noqa: F841
        sdt_results,
        prob_results,
        rt_by_block,
        trend_results,
        diagnostic_indices,
        result_dir,
    )

    # 10. 生成报告
    results = generate_report(
        trials_df,
        sdt_results,
        prob_results,
        rt_by_block,
        trend_results,
        diagnostic_indices,
        clinical_interpretation,
        result_dir,
    )

    # 记录结束时间
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n分析完成，耗时: {duration:.2f}秒")

    return results


def run_prt_analysis(cfg: DictConfig = None, data_utils: DataUtils = None):
    print("=" * 60)
    print("PRT（概率性奖励任务）分析")
    print("=" * 60)

    if data_utils is None:
        file_input = input("请输入数据文件路径: \n").strip("'").strip()
        file_path = Path(file_input.strip("'").strip('"')).resolve()
    else:
        file_path = (
            Path(cfg.output_dir) / data_utils.date / f"{data_utils.session_id}-prt.csv"
        )

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    print(f"正在读取数据文件: {file_path}")
    df = pl.read_csv(file_path)

    if cfg is None:
        result_dir = file_path.parent.parent / "results"
    else:
        result_dir = Path(cfg.result_dir)

    if data_utils is not None:
        result_dir = result_dir / str(data_utils.session_id)
    result_dir = result_dir / "prt_analysis"

    result_dir.mkdir(parents=True, exist_ok=True)

    results = analyze_prt_data(df=df, target_blocks=[0, 1, 2], result_dir=result_dir)

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_prt_analysis()
