import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from omegaconf import DictConfig
from plotly.subplots import make_subplots
from scipy import stats

from psycho.analysis.utils import (
    DataUtils,
    calculate_sample_size,
    check_normality_and_homoscedasticity,
    extract_trials_by_block,
    find_exp_files,
    perform_group_comparisons,
)

warnings.filterwarnings("ignore")


REFERENCE_VALUES = {
    "control": {
        "log_b": [0.19, 0.24, 0.23],
        "rich_hit_rate": 0.88,
        "lean_hit_rate": 0.75,
        "lean_miss_after_rewarded_rich": 0.26,
        "lean_miss_after_nonrewarded_rich": 0.23,
        "rich_miss_after_rewarded_rich": 0.13,
        "rich_miss_after_rewarded_lean": 0.11,
    },
    "mdd": {
        "log_b": [0.08, 0.12, 0.10],
        "rich_hit_rate": 0.86,
        "lean_hit_rate": 0.77,
        "lean_miss_after_rewarded_rich": 0.25,
        "lean_miss_after_nonrewarded_rich": 0.15,
        "rich_miss_after_rewarded_rich": 0.11,
        "rich_miss_after_rewarded_lean": 0.16,
    },
}
key_metrics = [
    "mean_log_b",
    "mean_hit_rate_diff",
    "mean_rt_diff",
]


def find_prt_files(data_dir: Path) -> list[Path]:
    """查找指定目录下的PRT实验结果文件"""
    EXP_TYPE = "prt"
    return find_exp_files(data_dir, EXP_TYPE)


def load_and_preprocess_data(df: pl.DataFrame) -> pl.DataFrame:
    """加载并预处理数据"""
    try:
        trials_df = extract_trials_by_block(
            df,
            target_block_indices=[0, 1, 2],
            block_col="block_index",
            trial_col="trial_index",
        )

        if trials_df.height == 0:
            print("❌ 错误: 未找到有效的试次数据")
            return None

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
    """识别每个Block的Rich刺激"""
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
    return rich_stim_results


def calculate_sdt_metrics(
    trials_df: pl.DataFrame, rich_stim_results: dict[int, dict[str, Any]]
) -> dict[int, dict[str, float]]:
    """计算SDT指标（反应偏向和辨别力）"""
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

        rich_total = rich_hit + rich_miss
        lean_total = lean_hit + lean_miss

        rich_hit_rate = rich_hit / rich_total if rich_total > 0 else 0
        lean_hit_rate = lean_hit / lean_total if lean_total > 0 else 0

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

    return sdt_results


def calculate_probability_analysis(
    trials_df: pl.DataFrame, rich_stim_results: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    """概率分析"""
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

    return prob_results


def analyze_reaction_time(
    trials_df: pl.DataFrame, rich_stim_results: dict[int, dict[str, Any]]
) -> dict[int, dict[str, float]]:
    """分析反应时"""
    rt_by_block = {}

    for block in sorted(trials_df["block_index"].unique()):
        block_data = trials_df.filter(pl.col("block_index") == block)
        rich_stim = rich_stim_results[block]["rich_stim"]

        block_data = block_data.with_columns(
            pl.when(pl.col("rt") >= 0.1)
            .then(pl.col("rt"))
            .otherwise(None)
            .alias("rt_clean")
        )

        rt_rich = block_data.filter(
            (pl.col("stim") == rich_stim) & (pl.col("rt_clean").is_not_null())
        )["rt_clean"].mean()

        rt_lean = block_data.filter(
            (pl.col("stim") != rich_stim) & (pl.col("rt_clean").is_not_null())
        )["rt_clean"].mean()

        rt_correct = block_data.filter(
            (pl.col("correct")) & (pl.col("rt_clean").is_not_null())
        )["rt_clean"].mean()
        rt_error = block_data.filter(
            (~pl.col("correct")) & (pl.col("rt_clean").is_not_null())
        )["rt_clean"].mean()

        rt_by_block[block] = {
            "rt_rich": rt_rich if rt_rich is not None else 0,
            "rt_lean": rt_lean if rt_lean is not None else 0,
            "rt_diff": (rt_lean if rt_lean is not None else 0)
            - (rt_rich if rt_rich is not None else 0),
            "rt_correct": rt_correct if rt_correct is not None else 0,
            "rt_error": rt_error if rt_error is not None else 0,
        }

    return rt_by_block


def analyze_performance_trends(trials_df: pl.DataFrame) -> dict[int, dict[str, Any]]:
    """表现趋势分析"""
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

            # 抛弃<0.1秒的反应时
            early_rt_clean = early_trials.filter(pl.col("rt") >= 0.1)["rt"]
            late_rt_clean = late_trials.filter(pl.col("rt") >= 0.1)["rt"]

            early_rt = early_rt_clean.mean() if early_rt_clean.shape[0] > 0 else None
            late_rt = late_rt_clean.mean() if late_rt_clean.shape[0] > 0 else None

            results[block] = {
                "early_accuracy": early_accuracy,
                "late_accuracy": late_accuracy,
                "accuracy_change": late_accuracy - early_accuracy,
                "early_rt": early_rt,
                "late_rt": late_rt,
                "rt_change": late_rt - early_rt
                if late_rt is not None and early_rt is not None
                else None,
            }

    return results


def calculate_key_metrics(
    sdt_results: dict[int, dict[str, float]],
    prob_results: dict[int, dict[str, Any]],
    rt_by_block: dict[int, dict[str, float]],
) -> dict[str, float]:
    """计算关键指标"""
    blocks = sorted(sdt_results.keys())

    mean_log_b = np.mean([sdt_results[b]["log_b"] for b in blocks])

    mean_log_d = np.mean([sdt_results[b]["log_d"] for b in blocks])

    mean_rich_hit_rate = np.mean([sdt_results[b]["rich_hit_rate"] for b in blocks])

    mean_lean_hit_rate = np.mean([sdt_results[b]["lean_hit_rate"] for b in blocks])

    # rich 击中率与 lean 击中率的差异, 越高, 被试越偏向Rich刺激
    mean_hit_rate_diff = mean_rich_hit_rate - mean_lean_hit_rate

    mean_rt_diff = np.mean([rt_by_block[b]["rt_diff"] for b in blocks])

    return {
        "mean_log_b": mean_log_b,
        "mean_log_d": mean_log_d,
        "mean_rich_hit_rate": mean_rich_hit_rate,
        "mean_lean_hit_rate": mean_lean_hit_rate,
        "mean_hit_rate_diff": mean_hit_rate_diff,
        "mean_rt_diff": mean_rt_diff,
    }


def create_visualizations(
    sdt_results: dict[int, dict[str, float]],
    prob_results: dict[int, dict[str, Any]],
    rt_by_block: dict[int, dict[str, float]],
    trend_results: dict[int, dict[str, Any]],
    key_metrics: dict[str, float],
    result_dir: Path,
) -> go.Figure:
    """创建可视化图表"""
    blocks = sorted(sdt_results.keys())

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
            "7. 关键指标总结",
            "8. 与参考对比",
            "9. 学习曲线",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "table"}, {"type": "bar"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.15,
    )

    # 图1: 反应偏向(Log b)随Block变化
    log_b_values = [sdt_results[b]["log_b"] for b in blocks]

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
            y=REFERENCE_VALUES["mdd"]["log_b"],
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

    # 添加参考线
    fig.add_hline(
        y=REFERENCE_VALUES["control"]["rich_hit_rate"],
        annotation_text="HC Rich",
        annotation_position="right",
        line=dict(width=1, dash="dash", color="green"),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=REFERENCE_VALUES["control"]["lean_hit_rate"],
        annotation_text="HC Lean",
        annotation_position="right",
        line=dict(width=1, dash="dash", color="darkgreen"),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=REFERENCE_VALUES["mdd"]["rich_hit_rate"],
        annotation_text="MDD Rich",
        annotation_position="right",
        line=dict(width=1, dash="dash", color="red"),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=REFERENCE_VALUES["mdd"]["lean_hit_rate"],
        annotation_text="MDD Lean",
        annotation_position="right",
        line=dict(width=1, dash="dash", color="darkred"),
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

    fig.add_trace(
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

    fig.add_trace(
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

    # 图7: 关键指标总结表格
    metrics_table = go.Table(
        header=dict(
            values=["指标", "值", "解释"], fill_color="lightblue", align="left"
        ),
        cells=dict(
            values=[
                [
                    "平均Log b",
                    "平均Log d",
                    "Rich击中率",
                    "Lean击中率",
                    "击中率差异",
                    "反应时差异",
                ],
                [
                    f"{key_metrics['mean_log_b']:.3f}",
                    f"{key_metrics['mean_log_d']:.3f}",
                    f"{key_metrics['mean_rich_hit_rate']:.3f}",
                    f"{key_metrics['mean_lean_hit_rate']:.3f}",
                    f"{key_metrics['mean_hit_rate_diff']:.3f}",
                    f"{key_metrics['mean_rt_diff']:.3f}",
                ],
                [
                    "反应偏向",
                    "辨别力",
                    "富刺激表现",
                    "贫刺激表现",
                    "表现差异",
                    "反应选择",
                ],
            ]
        ),
    )

    fig.add_trace(metrics_table, row=3, col=1)

    # 图8: 与参考对比
    fig.add_trace(
        go.Bar(
            x=["Log b", "击中率差异", "反应时差异"],
            y=[
                key_metrics["mean_log_b"],
                key_metrics["mean_hit_rate_diff"],
                key_metrics["mean_rt_diff"],
            ],
            name="当前被试",
            marker_color="blue",
        ),
        row=3,
        col=2,
    )

    # 添加文献参考值
    control_values = [
        np.mean(REFERENCE_VALUES["control"]["log_b"]),
        REFERENCE_VALUES["control"]["rich_hit_rate"]
        - REFERENCE_VALUES["control"]["lean_hit_rate"],
        0,  # 反应时差异参考值设为0
    ]

    mdd_values = [
        np.mean(REFERENCE_VALUES["mdd"]["log_b"]),
        REFERENCE_VALUES["mdd"]["rich_hit_rate"]
        - REFERENCE_VALUES["mdd"]["lean_hit_rate"],
        0,  # 反应时差异参考值设为0
    ]

    fig.add_trace(
        go.Scatter(
            x=["Log b", "击中率差异", "反应时差异"],
            y=control_values,
            mode="markers",
            name="文献对照组",
            marker=dict(size=12, color="green", symbol="square"),
            opacity=0.7,
        ),
        row=3,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=["Log b", "击中率差异", "反应时差异"],
            y=mdd_values,
            mode="markers",
            name="文献MDD组",
            marker=dict(size=12, color="red", symbol="diamond"),
            opacity=0.7,
        ),
        row=3,
        col=2,
    )

    # 图9: 学习曲线（以Block 0为例）
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
            col=3,
        )

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

    fig.update_xaxes(title_text="学习阶段", row=3, col=3)
    fig.update_yaxes(title_text="准确率", range=[0.5, 1.0], row=3, col=3)

    title_text = "PRT分析报告"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20, family="Arial Black"), x=0.5),
        height=1400,
        width=1600,
        showlegend=True,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    html_path = result_dir / "prt_visualization.html"

    fig.write_html(str(html_path))

    return fig


def save_results(
    sdt_results: dict[int, dict[str, float]],
    prob_results: dict[int, dict[str, Any]],
    rt_by_block: dict[int, dict[str, float]],
    key_metrics: dict[str, float],
    result_dir: Path,
):
    """保存结果"""

    blocks = sorted(sdt_results.keys())
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

    metrics_df = pl.DataFrame([key_metrics])
    metrics_df.write_csv(result_dir / "prt_key_metrics.csv")


def generate_report(
    trials_df: pl.DataFrame,
    sdt_results: dict[int, dict[str, float]],
    prob_results: dict[int, dict[str, Any]],
    rt_by_block: dict[int, dict[str, float]],
    trend_results: dict[int, dict[str, Any]],
    key_metrics: dict[str, float],
    result_dir: Path,
) -> dict[str, Any]:
    """生成PRT数据分析报告"""

    blocks = sorted(sdt_results.keys())
    overall_accuracy = trials_df.filter(pl.col("correct")).height / trials_df.height

    # 抛弃反应过快的数据(<0.1)
    valid_rt = trials_df.filter(pl.col("rt") >= 0.1)["rt"]
    mean_rt = valid_rt.mean() if valid_rt.shape[0] > 0 else None

    save_results(sdt_results, prob_results, rt_by_block, key_metrics, result_dir)

    print(f"\n结果已保存到: {result_dir}")
    print("  - prt_sdt_results.csv (SDT指标)")
    print("  - prt_probability_results.csv (概率分析结果)")
    print("  - prt_reaction_time_results.csv (反应时结果)")
    print("  - prt_key_metrics.csv (关键指标)")
    print("  - prt_visualization.html (主分析图表)")

    return {
        "data_summary": {
            "total_trials": trials_df.height,
            "num_blocks": len(blocks),
            "overall_accuracy": float(overall_accuracy),
            "mean_rt": float(mean_rt) if mean_rt is not None else None,
        },
        "key_metrics": key_metrics,
        "sdt_results": sdt_results,
        "prob_results": prob_results,
        "rt_by_block": rt_by_block,
    }


def analyze_prt_data(
    df: pl.DataFrame,
    target_blocks: list[int] = [0, 1, 2],
    result_dir: Path = Path("results"),
) -> dict[str, Any]:
    """分析单个被试的PRT数据"""

    # 1. 提取试次数据
    trials_df = load_and_preprocess_data(df)

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

    # 7. 计算关键指标
    key_metrics = calculate_key_metrics(sdt_results, prob_results, rt_by_block)

    # 8. 创建可视化
    fig = create_visualizations(  # noqa: F841
        sdt_results,
        prob_results,
        rt_by_block,
        trend_results,
        key_metrics,
        result_dir,
    )

    # 9. 生成报告
    results = generate_report(
        trials_df,
        sdt_results,
        prob_results,
        rt_by_block,
        trend_results,
        key_metrics,
        result_dir,
    )

    # 10. 保存结果
    save_results(sdt_results, prob_results, rt_by_block, key_metrics, result_dir)

    return results


def create_group_comparison_visualizations_single_group(
    group_metrics: list[dict[str, float]],
    statistical_results: dict[str, dict[str, Any]],
    result_dir: Path,
):
    """单个组的组分析可视化"""

    all_metrics = group_metrics

    key_metrics_list = [
        "mean_log_b",
        "mean_hit_rate_diff",
        "mean_rt_diff",
    ]

    metric_names = ["反应偏向(Log b)", "击中率差异", "反应时差异"]

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "1. 各被试反应偏向分布",
            "2. 关键指标相关性",
            "3. 对比参考值",
            "4. 统计检验结果",
            "5. 效应量分析",
            "6. 指标分布箱形图",
            "7. 样本量计算",
            "8. 样本量需求曲线",
            "9. 效应量分布",
        ),
        specs=[
            [{"type": "bar"}, {"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "table"}, {"type": "bar"}, {"type": "box"}],
            [{"type": "table"}, {"type": "scatter"}, {"type": "histogram"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # 图1: 各被试反应偏向分布
    log_b_values = [m["mean_log_b"] for m in all_metrics]
    subjects = [f"被试{i + 1}" for i in range(len(all_metrics))]

    fig.add_trace(
        go.Bar(
            x=subjects,
            y=log_b_values,
            name="反应偏向",
            marker_color="lightblue",
            text=[f"{v:.3f}" for v in log_b_values],
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # 图2: 关键指标相关性热图
    metrics_df = pd.DataFrame(all_metrics)
    corr_matrix = metrics_df[key_metrics_list].corr()

    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=metric_names,
            y=metric_names,
            colorscale="RdBu",
            zmid=0,
            text=np.around(corr_matrix.values, 2),
            texttemplate="%{text}",
            showscale=True,
        ),
        row=1,
        col=2,
    )
    fig.update_traces(
        colorbar=dict(len=0.25, y=0.5),
        selector=dict(type="heatmap"),
    )

    # 图3: 对比参考
    control_ref = np.mean(REFERENCE_VALUES["control"]["log_b"])
    mdd_ref = np.mean(REFERENCE_VALUES["mdd"]["log_b"])
    group_mean = np.mean(log_b_values)

    fig.add_trace(
        go.Bar(
            x=["当前组", "文献对照组", "文献MDD组"],
            y=[group_mean, control_ref, mdd_ref],
            name="反应偏向比较",
            marker_color=["blue", "green", "red"],
            text=[f"{group_mean:.3f}", f"{control_ref:.3f}", f"{mdd_ref:.3f}"],
            textposition="auto",
        ),
        row=1,
        col=3,
    )

    # 图4: 统计检验结果表格
    if "error" not in statistical_results:
        test_data = []
        for metric, name in zip(key_metrics_list, metric_names):
            if metric in statistical_results:
                result = statistical_results[metric]
                test_data.append(
                    [
                        name,
                        f"{result.get('t_statistic', result.get('statistic', 'N/A')):.3f}",
                        f"{result.get('p_value', 'N/A'):.4f}",
                        f"{result.get('cohens_d', 'N/A')}",
                        f"{result.get('effect_size_desc', 'N/A')}",
                    ]
                )

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["指标", "t值", "p值", "效应量", "效应大小"],
                    fill_color="lightblue",
                    align="left",
                    font=dict(size=10),
                ),
                cells=dict(
                    values=np.array(test_data).T if test_data else [[]],
                    fill_color="lavender",
                    align="left",
                    font=dict(size=9),
                ),
                columnwidth=[0.2, 0.15, 0.15, 0.15, 0.3],
            ),
            row=2,
            col=1,
        )

    # 图5: 效应量分析
    if "error" not in statistical_results:
        effect_sizes = []
        metrics_names = []

        for metric, name in zip(key_metrics_list, metric_names):
            if metric in statistical_results:
                result = statistical_results[metric]
                if "cohens_d" in result and result["cohens_d"] is not None:
                    effect_sizes.append(abs(result["cohens_d"]))
                    metrics_names.append(name.split("(")[0].strip())

        if effect_sizes:
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=effect_sizes,
                    name="效应量(绝对值)",
                    marker_color="lightgreen",
                    text=[f"{v:.2f}" for v in effect_sizes],
                    textposition="auto",
                ),
                row=2,
                col=2,
            )

    # 图6: 指标分布箱形图
    for i, (metric, name) in enumerate(zip(key_metrics_list, metric_names)):
        values = [m[metric] for m in all_metrics]
        fig.add_trace(
            go.Box(
                y=values,
                name=name,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                marker_color="lightblue",
                showlegend=False,
            ),
            row=2,
            col=3,
        )

    # 图7: 样本量计算表格
    if "error" not in statistical_results:
        sample_size_data = []
        for metric, name in zip(key_metrics_list, metric_names):
            if metric in statistical_results:
                result = statistical_results[metric]
                if (
                    "required_sample_size_per_group" in result
                    and result["required_sample_size_per_group"]
                ):
                    sample_size_data.append(
                        [
                            name,
                            f"{result.get('cohens_d', 'N/A'):.3f}",
                            f"{result['required_sample_size_per_group']}",
                            f"{result['required_total_sample_size'] if result.get('required_total_sample_size') else 'N/A'}",
                        ]
                    )

        if sample_size_data:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["指标", "效应量(d)", "每组需样本量", "总需样本量"],
                        fill_color="lightcoral",
                        align="left",
                        font=dict(size=10),
                    ),
                    cells=dict(
                        values=np.array(sample_size_data).T,
                        fill_color="mistyrose",
                        align="left",
                        font=dict(size=9),
                    ),
                ),
                row=3,
                col=1,
            )

    # 图8: 样本量需求曲线
    if "error" not in statistical_results and "mean_log_b" in statistical_results:
        effect_sizes = np.linspace(0.1, 1.0, 20)
        sample_sizes = []

        for d in effect_sizes:
            sample_size_info = calculate_sample_size(
                effect_size=d, alpha=0.05, power=0.8, test_type="two_sample"
            )
            sample_sizes.append(sample_size_info["required_n"])

        # 获取当前效应量
        current_d = statistical_results["mean_log_b"].get("cohens_d")

        fig.add_trace(
            go.Scatter(
                x=effect_sizes,
                y=sample_sizes,
                mode="lines",
                name="样本量需求曲线",
                line=dict(width=3, color="blue"),
                fill="tozeroy",
                fillcolor="rgba(0, 0, 255, 0.1)",
            ),
            row=3,
            col=2,
        )

        if current_d is not None:
            current_sample_size = calculate_sample_size(
                effect_size=abs(current_d),
                alpha=0.05,
                power=0.8,
                test_type="two_sample",
            )["required_n"]

            fig.add_trace(
                go.Scatter(
                    x=[abs(current_d)],
                    y=[current_sample_size],
                    mode="markers+text",
                    name="当前效应量",
                    marker=dict(size=15, color="red", symbol="diamond"),
                    text=[f"d={abs(current_d):.2f}<br>n={current_sample_size}"],
                    textposition="top center",
                ),
                row=3,
                col=2,
            )

        fig.update_xaxes(title_text="效应量 (Cohen's d)", row=3, col=2)
        fig.update_yaxes(title_text="每组所需样本量", row=3, col=2)

    # 图9: 效应量分布直方图
    if "error" not in statistical_results:
        cohens_d_values = []
        for metric in key_metrics_list:
            if metric in statistical_results:
                d = statistical_results[metric].get("cohens_d")
                if d is not None:
                    cohens_d_values.append(abs(d))

        if cohens_d_values:
            fig.add_trace(
                go.Histogram(
                    x=cohens_d_values,
                    nbinsx=10,
                    name="效应量分布",
                    marker_color="lightblue",
                    opacity=0.7,
                ),
                row=3,
                col=3,
            )

            fig.update_xaxes(title_text="效应量 (Cohen's d)", row=3, col=3)
            fig.update_yaxes(title_text="频数", row=3, col=3)

    fig.update_layout(
        title=dict(
            text="PRT组分析报告",
            font=dict(size=22, family="Arial Black"),
            x=0.5,
        ),
        height=1400,
        width=1600,
        showlegend=True,
        template="plotly_white",
    )

    # 保存图表
    fig.write_html(str(result_dir / "prt_group_analysis_report.html"))


def create_group_comparison_visualizations(
    control_metrics: list[dict[str, float]],
    experimental_metrics: list[dict[str, float]],
    comparison_results: dict[str, dict[str, Any]],
    result_dir: Path,
):
    control_values = {k: [] for k in control_metrics[0].keys()}
    experimental_values = {k: [] for k in experimental_metrics[0].keys()}

    for metric in control_metrics[0].keys():
        control_values[metric] = [m[metric] for m in control_metrics]
        experimental_values[metric] = [m[metric] for m in experimental_metrics]

    # 创建图表
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "1. 反应偏向(Log b)分布",
            "2. 击中率差异分布",
            "3. 反应时差异分布",
            "4. 关键指标对比",
            "5. 统计检验结果",
            "6. 效应量分析",
            "7. 样本量计算",
            "8. 样本量需求曲线",
            "9. 效应量与样本量关系",
        ),
        specs=[
            [{"type": "box"}, {"type": "box"}, {"type": "box"}],
            [{"type": "bar"}, {"type": "table"}, {"type": "bar"}],
            [{"type": "table"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.2,
    )

    # 图1-3: 指标分布箱形图
    key_metrics = ["mean_log_b", "mean_hit_rate_diff", "mean_rt_diff"]
    metric_names = ["反应偏向(Log b)", "击中率差异", "反应时差异"]

    for i, (metric, name) in enumerate(zip(key_metrics, metric_names)):
        fig.add_trace(
            go.Box(
                y=control_values[metric],
                name="对照组",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                marker_color="lightgreen",
                showlegend=(i == 0),
            ),
            row=1,
            col=i + 1,
        )

        fig.add_trace(
            go.Box(
                y=experimental_values[metric],
                name="实验组",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                marker_color="lightcoral",
                showlegend=(i == 0),
            ),
            row=1,
            col=i + 1,
        )

    # 图4: 关键指标对比
    control_means = [np.mean(control_values[metric]) for metric in key_metrics]
    control_stds = [np.std(control_values[metric], ddof=1) for metric in key_metrics]
    experimental_means = [
        np.mean(experimental_values[metric]) for metric in key_metrics
    ]
    experimental_stds = [
        np.std(experimental_values[metric], ddof=1) for metric in key_metrics
    ]

    x_positions = np.arange(len(key_metrics))

    fig.add_trace(
        go.Bar(
            x=x_positions - 0.2,
            y=control_means,
            name="对照组",
            marker_color="green",
            error_y=dict(type="data", array=control_stds, visible=True),
            width=0.4,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=x_positions + 0.2,
            y=experimental_means,
            name="实验组",
            marker_color="red",
            error_y=dict(type="data", array=experimental_stds, visible=True),
            width=0.4,
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(ticktext=metric_names, tickvals=x_positions, row=2, col=1)

    # 图5: 统计检验结果
    if comparison_results:
        table_data = []
        for metric in key_metrics:
            if metric in comparison_results:
                result = comparison_results[metric]

                # Safe formatting for optional values (Fixed)
                stat_val = result.get("statistic")
                p_val = result.get("p_value")
                # Try new key 'effect_size', fallback to old 'cohens_d' if needed
                eff_size = (
                    result.get("effect_size")
                    if result.get("effect_size") is not None
                    else result.get("cohens_d")
                )
                eff_mag = result.get(
                    "effect_size_magnitude", result.get("effect_size_desc", "N/A")
                )

                table_data.append(
                    [
                        metric_names[key_metrics.index(metric)],
                        str(
                            result.get("analysis_type", result.get("test_type", "N/A"))
                        ),
                        f"{stat_val:.3f}" if stat_val is not None else "N/A",
                        f"{p_val:.4f}" if p_val is not None else "N/A",
                        f"{eff_size:.3f}" if eff_size is not None else "N/A",
                        str(eff_mag),
                    ]
                )

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["指标", "检验方法", "统计量", "p值", "效应量", "效应大小"],
                    fill_color="lightblue",
                    align="left",
                    font=dict(size=10),
                ),
                cells=dict(
                    values=np.array(table_data).T if table_data else [[]],
                    fill_color="lavender",
                    align="left",
                    font=dict(size=9),
                ),
            ),
            row=2,
            col=2,
        )

    # 图6: 效应量分析
    if comparison_results:
        effect_sizes = []
        metric_labels = []

        for metric in key_metrics:
            if metric in comparison_results:
                # Try new key 'effect_size', fallback to old 'cohens_d'
                val = comparison_results[metric].get("effect_size")
                if val is None:
                    val = comparison_results[metric].get("cohens_d")

                if val is not None:
                    effect_sizes.append(abs(val))
                    metric_labels.append(metric_names[key_metrics.index(metric)])

        if effect_sizes:
            fig.add_trace(
                go.Bar(
                    x=metric_labels,
                    y=effect_sizes,
                    name="效应量(绝对值)",
                    marker_color="lightblue",
                    text=[f"{v:.2f}" for v in effect_sizes],
                    textposition="auto",
                ),
                row=2,
                col=3,
            )

    # 图7: 样本量计算表格
    if comparison_results:
        sample_size_data = []
        for metric, name in zip(key_metrics, metric_names):
            if metric in comparison_results:
                result = comparison_results[metric]
                if (
                    "required_sample_size_per_group" in result
                    and result["required_sample_size_per_group"]
                ):
                    sample_size_data.append(
                        [
                            name,
                            # f"{result.get('cohens_d', 'N/A'):.3f}",
                            f"{result['required_sample_size_per_group']}",
                            f"{result.get('required_total_sample_size', 'N/A')}",
                            f"{result.get('sample_size_power', 0.8):.2f}",
                            f"{result.get('sample_size_alpha', 0.05):.3f}",
                        ]
                    )

        if sample_size_data:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=[
                            "指标",
                            "效应量",
                            "每组需样本",
                            "总需样本",
                            "功效",
                            "α水平",
                        ],
                        fill_color="lightcoral",
                        align="left",
                        font=dict(size=9),
                    ),
                    cells=dict(
                        values=np.array(sample_size_data).T,
                        fill_color="mistyrose",
                        align="left",
                        font=dict(size=8),
                    ),
                ),
                row=3,
                col=1,
            )

    # 图8: 样本量需求曲线
    if comparison_results:
        effect_sizes = np.linspace(0.1, 1.0, 20)
        sample_sizes_per_group = []
        sample_sizes_total = []

        for d in effect_sizes:
            sample_size_info = calculate_sample_size(
                effect_size=d, alpha=0.05, power=0.8, test_type="two_sample"
            )
            sample_sizes_per_group.append(sample_size_info["required_n"])
            sample_sizes_total.append(sample_size_info["required_n_total"])

        first_metric = key_metrics[0]
        if first_metric in comparison_results:
            current_d = comparison_results[first_metric].get("cohens_d")

            fig.add_trace(
                go.Scatter(
                    x=effect_sizes,
                    y=sample_sizes_per_group,
                    mode="lines",
                    name="每组样本量",
                    line=dict(width=3, color="blue"),
                ),
                row=3,
                col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=effect_sizes,
                    y=sample_sizes_total,
                    mode="lines",
                    name="总样本量",
                    line=dict(width=3, color="red", dash="dash"),
                ),
                row=3,
                col=2,
            )

            if current_d is not None:
                current_sample_size = calculate_sample_size(
                    effect_size=abs(current_d),
                    alpha=0.05,
                    power=0.8,
                    test_type="two_sample",
                )

                fig.add_trace(
                    go.Scatter(
                        x=[abs(current_d)],
                        y=[current_sample_size["required_n"]],
                        mode="markers+text",
                        name="当前效应量",
                        marker=dict(size=15, color="green", symbol="diamond"),
                        text=[
                            f"d={abs(current_d):.2f}<br>n={current_sample_size['required_n']}"
                        ],
                        textposition="top center",
                    ),
                    row=3,
                    col=2,
                )

            fig.update_xaxes(title_text="效应量 (Cohen's d)", row=3, col=2)
            fig.update_yaxes(title_text="所需样本量", row=3, col=2)

    # 图9: 效应量与样本量关系（散点图）
    if comparison_results:
        d_values = []
        n_values = []
        metric_labels = []

        for metric, name in zip(key_metrics, metric_names):
            if metric in comparison_results:
                d = comparison_results[metric].get("cohens_d")
                if d is not None:
                    d_values.append(abs(d))
                    n_values.append(
                        comparison_results[metric].get(
                            "required_sample_size_per_group", 0
                        )
                    )
                    metric_labels.append(name)

        if d_values:
            fig.add_trace(
                go.Scatter(
                    x=d_values,
                    y=n_values,
                    mode="markers+text",
                    name="效应量vs样本量",
                    marker=dict(size=15, color="purple"),
                    text=metric_labels,
                    textposition="top center",
                ),
                row=3,
                col=3,
            )

            # 添加趋势线
            if len(d_values) > 1:
                z = np.polyfit(d_values, n_values, 2)
                p = np.poly1d(z)
                d_range = np.linspace(min(d_values), max(d_values), 50)
                n_fitted = p(d_range)

                fig.add_trace(
                    go.Scatter(
                        x=d_range,
                        y=n_fitted,
                        mode="lines",
                        name="趋势线",
                        line=dict(width=2, color="orange", dash="dash"),
                    ),
                    row=3,
                    col=3,
                )

            fig.update_xaxes(title_text="效应量 (Cohen's d)", row=3, col=3)
            fig.update_yaxes(title_text="每组所需样本量", row=3, col=3)

    fig.update_layout(
        title=dict(
            text="PRT组间比较分析报告",
            font=dict(size=22, family="Arial Black"),
            x=0.5,
        ),
        height=1400,
        width=1600,
        showlegend=True,
        template="plotly_white",
    )

    fig.write_html(str(result_dir / "prt_group_comparison_report.html"))


def run_single_prt_analysis(file_path: Path, result_dir: Path = None):
    """单个被试的PRT分析"""

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    if result_dir is None:
        result_dir = file_path.parent / "prt_results"

    result_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(file_path)

    result = analyze_prt_data(df=df, target_blocks=[0, 1, 2], result_dir=result_dir)

    print(f"\n✅ 分析完成！结果保存在: {result_dir}")
    return result


def run_group_prt_analysis(
    data_files: list[Path],
    result_dir: Path = None,
    reference_group: Literal["control", "mdd"] = None,
):
    """组PRT分析"""

    result_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    group_metrics = []

    for i, file_path in enumerate(data_files):
        print(f"分析被试 {i + 1}/{len(data_files)}: {file_path.name}")

        try:
            df = pl.read_csv(file_path)

            subject_id = file_path.stem.split("-")[0]

            subject_result_dir = result_dir / subject_id
            subject_result_dir.mkdir(parents=True, exist_ok=True)

            result = analyze_prt_data(
                df=df,
                target_blocks=[0, 1, 2],
                result_dir=subject_result_dir,
            )
            result["subject_id"] = subject_id

            if result:
                all_results.append(result)
                group_metrics.append(result["key_metrics"])
        except Exception as e:
            print(f"❌ 被试 {file_path.name} 分析出错: {e}")

    print(f"\n共完成 {len(all_results)}/{len(data_files)} 个被试的分析")

    if len(all_results) < 2:
        print("⚠️ 被试数量不足，无法进行组间统计检验")
        return {"all_results": all_results}

    if reference_group not in ["control", "mdd"]:
        print("\n请选择参考组:")
        print("1. 对照组 (control)")
        print("2. MDD组 (mdd)")
        choice = input("选择 (1/2): ").strip()

        reference_group = "control" if choice == "1" else "mdd"

    ref_values = REFERENCE_VALUES[reference_group]

    # 单样本t检验
    statistical_results = {}

    key_metrics = [
        "mean_log_b",
        "mean_hit_rate_diff",
        "mean_rt_diff",
    ]

    for metric in key_metrics:
        # 获取当前组的指标值
        group_values = [m[metric] for m in group_metrics]

        if len(group_values) < 2:
            statistical_results[metric] = {"error": "样本量不足"}
            continue

        if metric == "mean_log_b":
            ref_value = np.mean(ref_values["log_b"])
        elif metric == "mean_hit_rate_diff":
            ref_value = ref_values["rich_hit_rate"] - ref_values["lean_hit_rate"]
        elif metric == "mean_rt_diff":
            ref_value = 0  # 假设对照组和MDD组没有反应时差异
        else:
            continue

        t_stat, p_value = stats.ttest_1samp(group_values, ref_value)

        # 计算效应量（Cohen's d）
        mean_diff = np.mean(group_values) - ref_value
        std_group = np.std(group_values, ddof=1)
        cohens_d = mean_diff / std_group if std_group > 0 else 0

        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            size = "很小"
        elif abs_d < 0.5:
            size = "小"
        elif abs_d < 0.8:
            size = "中等"
        else:
            size = "大"

        effect_size_desc = f"{size} (d={cohens_d:.2f})"

        # 计算样本量
        sample_size_info = {}
        if cohens_d is not None:
            sample_size_info = calculate_sample_size(
                effect_size=abs(cohens_d),
                alpha=0.05,
                power=0.8,
                test_type="one_sample",  # 与参考值比较是单样本检验
            )

        statistical_results[metric] = {
            "group_mean": float(np.mean(group_values)),
            "group_std": float(np.std(group_values, ddof=1)),
            "group_n": len(group_values),
            "reference_mean": float(ref_value),
            "reference_group": reference_group,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "effect_size_desc": effect_size_desc,
            "required_sample_size": sample_size_info.get("required_n")
            if sample_size_info
            else None,
            "sample_size_power": sample_size_info.get("power")
            if sample_size_info
            else None,
            "sample_size_alpha": sample_size_info.get("alpha")
            if sample_size_info
            else None,
        }

    all_metrics_df = pd.DataFrame([r["key_metrics"] for r in all_results])
    all_metrics_df.insert(0, "subject_id", [r["subject_id"] for r in all_results])
    all_metrics_df.to_csv(result_dir / "group_all_metrics.csv", index=False)

    group_mean_metrics = all_metrics_df.mean(numeric_only=True).to_dict()
    group_std_metrics = all_metrics_df.std(numeric_only=True).to_dict()

    stats_df = pd.DataFrame(
        [group_mean_metrics, group_std_metrics], index=["mean", "std"]
    ).T
    stats_df.to_csv(result_dir / "group_statistics.csv")

    sample_size_data = []
    for metric, result in statistical_results.items():
        if "required_sample_size" in result:
            sample_size_data.append(
                {
                    "metric": metric,
                    "effect_size": result.get("cohens_d"),
                    "required_sample_size": result.get("required_sample_size"),
                    "current_sample_size": result.get("group_n"),
                    "power": result.get("sample_size_power"),
                    "alpha": result.get("sample_size_alpha"),
                    "effect_size_magnitude": result.get("effect_size_desc", "")
                    .split("(")[0]
                    .strip(),
                }
            )

    if sample_size_data:
        sample_size_df = pd.DataFrame(sample_size_data)
        sample_size_df.to_csv(result_dir / "sample_size_calculations.csv", index=False)

    if "error" not in statistical_results:
        stats_test_df = pd.DataFrame(statistical_results).T
        stats_test_df.to_csv(result_dir / "group_statistical_tests.csv")

    create_group_comparison_visualizations_single_group(
        group_metrics, statistical_results, result_dir
    )

    print(f"\n✅ 组分析完成！结果保存在: {result_dir}")

    return {
        "all_results": all_results,
        "group_metrics": group_metrics,
        "statistical_results": statistical_results,
        "group_mean": group_mean_metrics,
        "group_std": group_std_metrics,
    }


def run_groups_prt_analysis(
    control_files: list[Path],
    experimental_files: list[Path],
    result_dir: Path = Path("group_comparison_results"),
    groups: list[str] = None,
) -> dict[str, Any]:
    """比较对照组和实验组"""

    control_results = []
    control_metrics = []
    control_name = groups[0]

    for i, file_path in enumerate(control_files):
        try:
            df = pl.read_csv(file_path)
            subject_id = file_path.stem.split("-")[0]

            subject_result_dir = result_dir / control_name / subject_id
            subject_result_dir.mkdir(parents=True, exist_ok=True)

            # 分析单个被试
            result = analyze_prt_data(
                df=df,
                target_blocks=[0, 1, 2],
                result_dir=subject_result_dir,
            )
            result["subject_id"] = subject_id

            if result:
                control_results.append(result)
                control_metrics.append(result["key_metrics"])

        except Exception as e:
            print(f"❌ 对照组被试 {file_path.name} 分析出错: {e}")

    experimental_results = []
    experimental_metrics = []
    experimental_name = groups[1]

    for i, file_path in enumerate(experimental_files):
        try:
            df = pl.read_csv(file_path)
            subject_id = file_path.stem.split("-")[0]

            subject_result_dir = result_dir / experimental_name / subject_id
            subject_result_dir.mkdir(parents=True, exist_ok=True)

            # 分析单个被试
            result = analyze_prt_data(
                df=df,
                target_blocks=[0, 1, 2],
                result_dir=subject_result_dir,
            )
            result["subject_id"] = subject_id

            if result:
                experimental_results.append(result)
                experimental_metrics.append(result["key_metrics"])

        except Exception as e:
            print(f"❌ 实验组被试 {file_path.name} 分析出错: {e}")

    if len(control_results) < 2 or len(experimental_results) < 2:
        print("⚠️ 任一组被试数量不足，无法进行组间统计检验")
        return {
            "control_results": control_results,
            "experimental_results": experimental_results,
            "control_metrics": control_metrics,
            "experimental_metrics": experimental_metrics,
        }

    print("\n检查正态性和方差齐性...")
    normality_results = check_normality_and_homoscedasticity(
        control_metrics + experimental_metrics, key_metrics
    )

    print("\n执行组间比较分析...")
    comparison_results = perform_group_comparisons(
        control_metrics, experimental_metrics, key_metrics
    )

    print("\n保存组分析结果...")

    # 保存所有被试的汇总指标
    all_control_metrics_df = pd.DataFrame([r["key_metrics"] for r in control_results])
    all_control_metrics_df.insert(0, "group", "control")
    all_control_metrics_df.insert(
        1, "subject_id", [r["subject_id"] for r in control_results]
    )

    all_experimental_metrics_df = pd.DataFrame(
        [r["key_metrics"] for r in experimental_results]
    )
    all_experimental_metrics_df.insert(0, "group", "experimental")
    all_experimental_metrics_df.insert(
        1, "subject_id", [r["subject_id"] for r in experimental_results]
    )

    all_metrics_df = pd.concat(
        [all_control_metrics_df, all_experimental_metrics_df], ignore_index=True
    )
    all_metrics_df.to_csv(result_dir / "all_subjects_metrics.csv", index=False)

    control_stats = all_control_metrics_df.drop(
        columns=["group", "subject_id"]
    ).describe()
    experimental_stats = all_experimental_metrics_df.drop(
        columns=["group", "subject_id"]
    ).describe()

    control_stats.to_csv(result_dir / "control_group_statistics.csv")
    experimental_stats.to_csv(result_dir / "experimental_group_statistics.csv")

    sample_size_data = []
    for metric, result in comparison_results.items():
        if "required_sample_size_per_group" in result:
            sample_size_data.append(
                {
                    "metric": metric,
                    "effect_size": result.get("cohens_d"),
                    "required_per_group": result.get("required_sample_size_per_group"),
                    "required_total": result.get("required_total_sample_size"),
                    "control_n": result.get("control_n"),
                    "experimental_n": result.get("experimental_n"),
                    "power": result.get("sample_size_power"),
                    "alpha": result.get("sample_size_alpha"),
                    "effect_size_magnitude": result.get("effect_size_magnitude", ""),
                }
            )

    if sample_size_data:
        sample_size_df = pd.DataFrame(sample_size_data)
        sample_size_df.to_csv(result_dir / "sample_size_calculations.csv", index=False)

    # 保存统计检验结果
    if normality_results:
        normality_df = pd.DataFrame(normality_results).T
        normality_df.to_csv(result_dir / "normality_tests.csv")

    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results).T
        comparison_df.to_csv(result_dir / "group_comparisons.csv")

    # 创建组比较可视化
    create_group_comparison_visualizations(
        control_metrics, experimental_metrics, comparison_results, result_dir
    )

    print(f"\n✅ 组间比较分析完成！结果保存在: {result_dir}")

    return {
        "control_results": control_results,
        "experimental_results": experimental_results,
        "control_metrics": control_metrics,
        "experimental_metrics": experimental_metrics,
        "normality_results": normality_results,
        "comparison_results": comparison_results,
    }


def run_prt_analysis(cfg: DictConfig = None, data_utils: DataUtils = None):
    """PRT（概率性奖励任务）分析入口函数"""

    if cfg is None:
        result_root = Path(
            input("请输入保存结果目录路径: ").strip("'").strip()
        ).resolve()
    else:
        result_root = Path(cfg.result_dir)
    if data_utils is None:
        print("\n请选择分析模式:")
        print("1. 单个被试分析")
        print("2. 组分析（多个被试）")
        print("3. 对照组 vs 实验组比较分析")

        choice = input("选择 (1/2/3): ").strip()

        if choice == "1":
            file_input = input("请输入数据文件路径: ").strip("'").strip()
            file_path = Path(file_input.strip("'").strip('"')).resolve()

            if not file_path.exists():
                print(f"❌ 文件不存在: {file_path}")
                return

            result_dir = result_root / "prt_results"
            result_dir.mkdir(parents=True, exist_ok=True)

            result = run_single_prt_analysis(file_path, result_dir)
            return result

        elif choice == "2":
            dir_input = input("请输入包含多个数据文件的目录路径: ").strip("'").strip()
            data_dir = Path(dir_input.strip("'").strip('"')).resolve()

            if not data_dir.exists():
                print(f"❌ 目录不存在: {data_dir}")
                return

            data_files = find_prt_files(data_dir)
            print(f"找到 {len(data_files)} 个数据文件")

            result_dir = result_root / "prt_group_results"
            result_dir.mkdir(parents=True, exist_ok=True)

            result = run_group_prt_analysis(data_files, result_dir)
            return result

        elif choice == "3":
            control_dir_input = (
                input("请输入对照组数据文件目录路径: ").strip("'").strip()
            )
            experimental_dir_input = (
                input("请输入实验组数据文件目录路径: ").strip("'").strip()
            )

            control_dir = Path(control_dir_input.strip("'").strip('"')).resolve()
            experimental_dir = Path(
                experimental_dir_input.strip("'").strip('"')
            ).resolve()

            if not control_dir.exists():
                print(f"❌ 对照组目录不存在: {control_dir}")
                return

            if not experimental_dir.exists():
                print(f"❌ 实验组目录不存在: {experimental_dir}")
                return

            control_files = find_prt_files(control_dir)
            experimental_files = find_prt_files(experimental_dir)

            print(
                f"找到 {len(control_files)} 个对照组文件和 {len(experimental_files)} 个实验组文件"
            )

            result_dir = result_root / "prt_group_comparison_results"
            result_dir.mkdir(parents=True, exist_ok=True)

            result = run_groups_prt_analysis(
                control_files, experimental_files, result_dir
            )

            return result

        else:
            print("❌ 无效的选择")
            return None
    else:
        if data_utils.session_id is None or (
            data_utils.groups is not None and len(data_utils.groups) > 0
        ):
            groups = data_utils.groups
            data_root = Path(cfg.output_dir)
            if len(groups) == 1:
                # 单个组分析
                files = find_prt_files(data_root / groups[0])
                result_dir = result_root / f"prt_{groups[0]}_results"
                result_dir.mkdir(parents=True, exist_ok=True)

                reference_group = "mdd" if "mdd" in groups[0] else "control"
                run_group_prt_analysis(
                    files, result_dir, reference_group=reference_group
                )
            else:
                # 多个组分析
                control_files = find_prt_files(data_root / groups[0])
                experimental_files = find_prt_files(data_root / groups[1])
                result_dir = (
                    result_root / f"prt_{groups[0]}_{groups[1]}_comparison_results"
                )
                result_dir.mkdir(parents=True, exist_ok=True)
                run_groups_prt_analysis(
                    control_files, experimental_files, result_dir, groups
                )
        else:
            # 单个被试分析
            file_path = (
                Path(cfg.output_dir)
                / data_utils.date
                / f"{data_utils.session_id}-prt.csv"
            )

            if not file_path.exists():
                print(f"❌ 文件不存在: {file_path}")
                return

            result_dir = result_root / str(data_utils.session_id) / "prt_analysis"
            result_dir.mkdir(parents=True, exist_ok=True)

            result = run_single_prt_analysis(file_path, result_dir)
            return result


if __name__ == "__main__":
    run_prt_analysis()
