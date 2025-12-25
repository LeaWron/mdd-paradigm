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
    create_common_comparison_figures,
    create_common_single_group_figures,
    extract_trials_by_block,
    find_exp_files,
    perform_group_comparisons,
    save_html_report,
)

warnings.filterwarnings("ignore")


REFERENCE_VALUES = {
    "control": {
        "log_b_block_0": 0.19,
        "log_b_block_1": 0.24,
        "log_b_block_2": 0.23,
        "rich_hit_rate": 0.88,
        "lean_hit_rate": 0.75,
        "lean_miss_after_rewarded_rich": 0.26,
        "lean_miss_after_nonrewarded_rich": 0.23,
        "rich_miss_after_rewarded_rich": 0.13,
        "rich_miss_after_rewarded_lean": 0.11,
    },
    "mdd": {
        "log_b_block_0": 0.08,
        "log_b_block_1": 0.12,
        "log_b_block_2": 0.10,
        "rich_hit_rate": 0.86,
        "lean_hit_rate": 0.77,
        "lean_miss_after_rewarded_rich": 0.25,
        "lean_miss_after_nonrewarded_rich": 0.15,
        "rich_miss_after_rewarded_rich": 0.11,
        "rich_miss_after_rewarded_lean": 0.16,
    },
}
key_metrics = [
    "log_b_block_0",
    "log_b_block_1",
    "log_b_block_2",
    "mean_rich_hit_rate",
    "mean_lean_hit_rate",
    "mean_lean_miss_after_rewarded_rich",
    "mean_lean_miss_after_nonrewarded_rich",
    "mean_rich_miss_after_rewarded_rich",
    "mean_rich_miss_after_rewarded_lean",
]

# 反应时是 贫-富
# 命中率是 富-贫
metric_names = [
    "反应偏向-Block 0",
    "反应偏向-Block 1",
    "反应偏向-Block 2",
    "富击中率",
    "贫击中率",
    "贫刺激错误率-前富奖励",
    "贫刺激错误率-前富未奖",
    "富刺激错误率-前富奖励",
    "富刺激错误率-前贫奖励",
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
        lean_trials_after_rewarded_rich = lean_trials.filter(
            (pl.col("prev_stim") == rich_stim) & (pl.col("prev_rewarded"))
        )

        # A2: 前一个试次是rich但无奖励
        lean_trials_after_nonrewarded_rich = lean_trials.filter(
            (pl.col("prev_stim") == rich_stim) & (~pl.col("prev_rewarded"))
        )

        lean_miss_rate_after_rewarded_rich = (
            (
                lean_trials_after_rewarded_rich.filter(~pl.col("correct")).height
                / lean_trials_after_rewarded_rich.height
            )
            if lean_trials_after_rewarded_rich.height > 0
            else 0
        )
        lean_miss_rate_after_nonrewarded_rich = (
            (
                lean_trials_after_nonrewarded_rich.filter(~pl.col("correct")).height
                / lean_trials_after_nonrewarded_rich.height
            )
            if lean_trials_after_nonrewarded_rich.height > 0
            else 0
        )

        # 情况B: 分析rich miss概率
        rich_trials = valid_data.filter(pl.col("stim") == rich_stim)

        # B1: 前一个试次是rich且获得奖励
        rich_trials_after_rewarded_rich = rich_trials.filter(
            (pl.col("prev_stim") == rich_stim) & (pl.col("prev_rewarded"))
        )

        # B2: 前一个试次是lean且获得奖励
        rich_trials_after_rewarded_lean = rich_trials.filter(
            (pl.col("prev_stim") == lean_stim) & (pl.col("prev_rewarded"))
        )

        rich_miss_rate_after_rewarded_rich = (
            (
                rich_trials_after_rewarded_rich.filter(~pl.col("correct")).height
                / rich_trials_after_rewarded_rich.height
            )
            if rich_trials_after_rewarded_rich.height > 0
            else 0
        )
        rich_miss_rate_after_rewarded_lean = (
            (
                rich_trials_after_rewarded_lean.filter(~pl.col("correct")).height
                / rich_trials_after_rewarded_lean.height
            )
            if rich_trials_after_rewarded_lean.height > 0
            else 0
        )

        prob_results[block] = {
            "lean_miss_after_rewarded_rich": lean_miss_rate_after_rewarded_rich,
            "lean_miss_after_nonrewarded_rich": lean_miss_rate_after_nonrewarded_rich,
            "rich_miss_after_rewarded_rich": rich_miss_rate_after_rewarded_rich,
            "rich_miss_after_rewarded_lean": rich_miss_rate_after_rewarded_lean,
            "counts": {
                "lean_miss_after_rewarded_rich": lean_trials_after_rewarded_rich.height,
                "lean_miss_after_nonrewarded_rich": lean_trials_after_nonrewarded_rich.height,
                "rich_miss_after_rewarded_rich": rich_trials_after_rewarded_rich.height,
                "rich_miss_after_rewarded_lean": rich_trials_after_rewarded_lean.height,
            },
            "miss_counts": {
                "lean_miss_after_rewarded_rich": lean_trials_after_rewarded_rich.filter(
                    ~pl.col("correct")
                ).height,
                "lean_miss_after_nonrewarded_rich": lean_trials_after_nonrewarded_rich.filter(
                    ~pl.col("correct")
                ).height,
                "rich_miss_after_rewarded_rich": rich_trials_after_rewarded_rich.filter(
                    ~pl.col("correct")
                ).height,
                "rich_miss_after_rewarded_lean": rich_trials_after_rewarded_lean.filter(
                    ~pl.col("correct")
                ).height,
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

    # mean_log_b = np.mean([sdt_results[b]["log_b"] for b in blocks])

    mean_log_d = np.mean([sdt_results[b]["log_d"] for b in blocks])

    mean_rich_hit_rate = np.mean([sdt_results[b]["rich_hit_rate"] for b in blocks])

    mean_lean_hit_rate = np.mean([sdt_results[b]["lean_hit_rate"] for b in blocks])

    # rich 击中率与 lean 击中率的差异, 越高, 被试越偏向Rich刺激
    mean_hit_rate_diff = mean_rich_hit_rate - mean_lean_hit_rate

    # lean 反应时与 rich 反应时的差异, 越高, 被试越偏向Rich刺激
    mean_rt_diff = np.mean([rt_by_block[b]["rt_diff"] for b in blocks])

    mean_lean_miss_after_rewarded_rich = np.sum(
        [
            prob_results[b]["miss_counts"]["lean_miss_after_rewarded_rich"]
            for b in blocks
        ]
    ) / np.sum(
        [prob_results[b]["counts"]["rich_miss_after_rewarded_rich"] for b in blocks]
    )

    mean_lean_miss_after_nonrewarded_rich = np.sum(
        [
            prob_results[b]["miss_counts"]["lean_miss_after_nonrewarded_rich"]
            for b in blocks
        ]
    ) / np.sum(
        [prob_results[b]["counts"]["lean_miss_after_nonrewarded_rich"] for b in blocks]
    )

    mean_rich_miss_after_rewarded_rich = np.sum(
        [
            prob_results[b]["miss_counts"]["rich_miss_after_rewarded_rich"]
            for b in blocks
        ]
    ) / np.sum(
        [prob_results[b]["counts"]["rich_miss_after_rewarded_rich"] for b in blocks]
    )

    mean_rich_miss_after_rewarded_lean = np.sum(
        [
            prob_results[b]["miss_counts"]["rich_miss_after_rewarded_lean"]
            for b in blocks
        ]
    ) / np.sum(
        [prob_results[b]["counts"]["rich_miss_after_rewarded_lean"] for b in blocks]
    )

    return_metrics = {}
    for b in blocks:
        return_metrics[f"log_b_block_{b}"] = sdt_results[b]["log_b"]
        # 如果你想看每个 block 的辨别力，也可以顺手加上：
        return_metrics[f"log_d_block_{b}"] = sdt_results[b]["log_d"]

    return_metrics.update(
        {
            "mean_log_d": mean_log_d,
            "mean_rich_hit_rate": mean_rich_hit_rate,
            "mean_lean_hit_rate": mean_lean_hit_rate,
            "mean_hit_rate_diff": mean_hit_rate_diff,
            "mean_rt_diff": mean_rt_diff,
            "mean_lean_miss_after_rewarded_rich": mean_lean_miss_after_rewarded_rich
            if not np.isnan(mean_lean_miss_after_rewarded_rich)
            else 0.0,
            "mean_lean_miss_after_nonrewarded_rich": mean_lean_miss_after_nonrewarded_rich
            if not np.isnan(mean_lean_miss_after_nonrewarded_rich)
            else 0.0,
            "mean_rich_miss_after_rewarded_rich": mean_rich_miss_after_rewarded_rich
            if not np.isnan(mean_rich_miss_after_rewarded_rich)
            else 0.0,
            "mean_rich_miss_after_rewarded_lean": mean_rich_miss_after_rewarded_lean
            if not np.isnan(mean_rich_miss_after_rewarded_lean)
            else 0.0,
        }
    )

    return return_metrics


def create_single_visualizations(
    sdt_results: dict[int, dict[str, float]],
    prob_results: dict[int, dict[str, Any]],
    rt_by_block: dict[int, dict[str, float]],
    trend_results: dict[int, dict[str, Any]],
    key_metrics: dict[str, float],
    result_dir: Path,
) -> list[go.Figure]:
    """创建可视化图表"""
    figs = []
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
            "8. 学习曲线",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "table"}, None, {"type": "scatter"}],
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
            y=[REFERENCE_VALUES["mdd"][f"log_b_block_{b}"] for b in blocks],
            mode="lines",
            name="MDD参考",
            line=dict(width=2, color="red", dash="dash"),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=blocks,
            y=[REFERENCE_VALUES["control"][f"log_b_block_{b}"] for b in blocks],
            mode="lines",
            name="对照组参考",
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
    # avg_lean_miss1 = np.mean(
    # [prob_results[b]["lean_miss_after_rewarded_rich"] for b in prob_results]
    # )
    # avg_lean_miss2 = np.mean(
    # [prob_results[b]["lean_miss_after_nonrewarded_rich"] for b in prob_results]
    # )
    mean_lean_miss_after_rewarded_rich = key_metrics[
        "mean_lean_miss_after_rewarded_rich"
    ]
    mean_lean_miss_after_nonrewarded_rich = key_metrics[
        "mean_lean_miss_after_nonrewarded_rich"
    ]

    fig.add_trace(
        go.Bar(
            x=["前试次富刺激有奖励", "前试次富刺激无奖励"],
            y=[
                mean_lean_miss_after_rewarded_rich,
                mean_lean_miss_after_nonrewarded_rich,
            ],
            name="Lean miss概率",
            marker_color=["royalblue", "crimson"],
            text=[
                f"{mean_lean_miss_after_rewarded_rich:.3f}",
                f"{mean_lean_miss_after_nonrewarded_rich:.3f}",
            ],
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
            name="MDD参考",
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
            name="对照组参考",
            marker=dict(size=12, color="green", symbol="diamond"),
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    # 图5: Rich miss概率分析
    # avg_rich_miss1 = np.mean(
    # [prob_results[b]["rich_miss_after_rewarded_rich"] for b in prob_results]
    # )
    # avg_rich_miss2 = np.mean(
    # [prob_results[b]["rich_miss_after_rewarded_lean"] for b in prob_results]
    # )
    mean_rich_miss_after_rewarded_rich = key_metrics[
        "mean_rich_miss_after_rewarded_rich"
    ]
    mean_rich_miss_after_rewarded_lean = key_metrics[
        "mean_rich_miss_after_rewarded_lean"
    ]

    fig.add_trace(
        go.Bar(
            x=["前试次富刺激有奖励", "前试次贫刺激有奖励"],
            y=[mean_rich_miss_after_rewarded_rich, mean_rich_miss_after_rewarded_lean],
            name="Rich miss概率",
            marker_color=["royalblue", "crimson"],
            text=[
                f"{mean_rich_miss_after_rewarded_rich:.3f}",
                f"{mean_rich_miss_after_rewarded_lean:.3f}",
            ],
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
            name="MDD参考",
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
            name="对照组参考",
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
                    "Log b-Block 0",
                    "Log b-Block 1",
                    "Log b-Block 2",
                    "平均Log d",
                    "Rich击中率",
                    "Lean击中率",
                    "击中率差异",
                ],
                [
                    f"{key_metrics['log_b_block_0']:.3f}",
                    f"{key_metrics['log_b_block_1']:.3f}",
                    f"{key_metrics['log_b_block_2']:.3f}",
                    f"{key_metrics['mean_log_d']:.3f}",
                    f"{key_metrics['mean_rich_hit_rate']:.3f}",
                    f"{key_metrics['mean_lean_hit_rate']:.3f}",
                    f"{key_metrics['mean_hit_rate_diff']:.3f}",
                ],
                [
                    "反应偏向",
                    "辨别力",
                    "",
                    "",
                    "富-贫",
                ],
            ]
        ),
    )

    fig.add_trace(metrics_table, row=3, col=1)

    # fig.add_trace(go.Bar(x=[], y=[], mode="lines", name=""), row=3, col=2)

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
    figs.append(fig)

    return figs


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

    # save_results(sdt_results, prob_results, rt_by_block, key_metrics, result_dir)

    print(f"\n✅ 分析完成! 结果保存在: {result_dir}")

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
    fig = create_single_visualizations(  # noqa: F841
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


def create_single_group_visualizations(
    group_metrics: list[dict[str, float]],
) -> list[go.Figure]:
    """单个组的组分析可视化"""

    all_metrics = group_metrics

    figs = []

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=(
            "1. 各被试反应偏向",
            "2. 各被试反应时差异",
            "3. 各被试条件性错误率",  # 修改标题
            "4. 各被试命中率对比",  # 新增子图
            "5. 反应偏向分布",
            "6. 反应时差异分布",
            "7. 条件性错误率分布",  # 修改标题
            "8. 命中率分布",  # 新增子图
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "box"}, {"type": "box"}, {"type": "box"}, {"type": "box"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # 对受试者ID进行排序
    subject_ids = [(int(m["subject_id"]), i) for i, m in enumerate(all_metrics)]
    subject_ids.sort()
    all_metrics = [all_metrics[m[1]] for m in subject_ids]
    subjects = [f"s{m[0]}" for m in subject_ids]

    # 图1: 各被试反应偏向
    blocks = list(range(3))
    log_b_values = []
    for i, _ in enumerate(blocks):
        log_b_values.append([])
        log_b_values[i] = [m[f"log_b_block_{i}"] for m in all_metrics]

    # 绘制反应偏向柱状图（分块）
    for i, log_bs in enumerate(log_b_values):
        fig.add_trace(
            go.Bar(
                x=subjects,
                y=log_bs,
                name=f"Block{i + 1}",
                text=[f"{v:.3f}" for v in log_bs],
                textposition="auto",
            ),
            row=1,
            col=1,
        )

    # 图2: 各被试反应时差异
    rt_diff_values = [m["mean_rt_diff"] for m in all_metrics]
    fig.add_trace(
        go.Bar(
            x=subjects,
            y=rt_diff_values,
            name="反应时差异",
            text=[f"{v:.3f}" for v in rt_diff_values],
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    # 图3: 各被试条件性错误率
    probs_keys = [
        "mean_lean_miss_after_rewarded_rich",
        "mean_lean_miss_after_nonrewarded_rich",
        "mean_rich_miss_after_rewarded_rich",
        "mean_rich_miss_after_rewarded_lean",
    ]
    probs_dicts = {key: [] for key in probs_keys}
    names = ["前富奖励贫", "前富不奖贫", "前富奖励富", "前贫奖励富"]

    for key in probs_keys:
        probs_dicts[key] = [m[key] for m in all_metrics]

    # 将4个条件性错误率堆叠显示
    for i, key in enumerate(probs_keys):
        fig.add_trace(
            go.Bar(
                x=subjects,
                y=probs_dicts[key],
                name=names[i],
                text=[f"{v:.3f}" for v in probs_dicts[key]],
                textposition="auto",
            ),
            row=1,
            col=3,
        )

    # 图4: 各被试命中率对比（新增）
    rich_hit_values = [m["mean_rich_hit_rate"] for m in all_metrics]
    lean_hit_values = [m["mean_lean_hit_rate"] for m in all_metrics]

    # 添加Rich命中率
    fig.add_trace(
        go.Bar(
            x=subjects,
            y=rich_hit_values,
            name="Rich命中率",
            text=[f"{v:.3f}" for v in rich_hit_values],
            textposition="auto",
        ),
        row=1,
        col=4,
    )

    # 添加Lean命中率
    fig.add_trace(
        go.Bar(
            x=subjects,
            y=lean_hit_values,
            name="Lean命中率",
            text=[f"{v:.3f}" for v in lean_hit_values],
            textposition="auto",
        ),
        row=1,
        col=4,
    )

    # 图5-8: 各指标的分布箱形图
    # 5. 反应偏向分布
    for i, log_bs in enumerate(log_b_values):
        fig.add_trace(
            go.Box(
                y=log_bs,
                boxmean="sd",
                name=f"Block{i + 1}",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            ),
            row=2,
            col=1,
        )

    # 6. 反应时差异分布
    fig.add_trace(
        go.Box(
            y=rt_diff_values,
            boxmean="sd",
            name="反应时差异",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        ),
        row=2,
        col=2,
    )

    # 7. 条件性错误率分布
    for i, key in enumerate(probs_keys):
        fig.add_trace(
            go.Box(
                y=probs_dicts[key],
                boxmean="sd",
                name=names[i],
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            ),
            row=2,
            col=3,
        )

    # 8. 命中率分布（新增）
    fig.add_trace(
        go.Box(
            y=rich_hit_values,
            boxmean="sd",
            name="Rich命中率",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        ),
        row=2,
        col=4,
    )
    fig.add_trace(
        go.Box(
            y=lean_hit_values,
            boxmean="sd",
            name="Lean命中率",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        ),
        row=2,
        col=4,
    )

    fig.update_layout(
        height=600 * 2,  # 增加高度以适应3行
        width=500 * 4,  # 增加宽度以适应4列
        showlegend=True,
        template="plotly_white",
    )

    figs.append(fig)
    return figs


def create_multi_group_visualizations(
    control_metrics: list[dict[str, float]],
    experimental_metrics: list[dict[str, float]],
):
    control_values = {k: [] for k in control_metrics[0].keys()}
    experimental_values = {k: [] for k in experimental_metrics[0].keys()}

    for metric in control_metrics[0].keys():
        control_values[metric] = [m[metric] for m in control_metrics]
        experimental_values[metric] = [m[metric] for m in experimental_metrics]

    # 创建图表 - 重新设计布局
    fig = make_subplots(
        rows=2,  # 2行
        cols=4,  # 4列
        subplot_titles=(
            "1. 反应偏向对比",
            "2. Rich命中率对比",
            "3. Lean命中率对比",
            "4. 条件性错误率对比",  # 修改标题
            "5. 反应时差异对比",
            "6. 反应偏向箱形图",
            "7. 命中率箱形图",
            "8. 条件性错误率箱形图",  # 新增子图
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "box"}, {"type": "box"}, {"type": "box"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # 1. 反应偏向对比（3个block分开显示）
    log_b_metrics = ["log_b_block_0", "log_b_block_1", "log_b_block_2"]

    # 为每个block创建单独的柱子
    x_positions = np.arange(len(log_b_metrics))

    # 准备数据
    control_means = [np.mean(control_values[metric]) for metric in log_b_metrics]
    control_stds = [np.std(control_values[metric], ddof=1) for metric in log_b_metrics]
    experimental_means = [
        np.mean(experimental_values[metric]) for metric in log_b_metrics
    ]
    experimental_stds = [
        np.std(experimental_values[metric], ddof=1) for metric in log_b_metrics
    ]

    # 添加对照组trace
    fig.add_trace(
        go.Bar(
            x=x_positions - 0.2,
            y=control_means,
            name="对照组",
            marker_color="green",
            error_y=dict(type="data", array=control_stds, visible=True),
            width=0.4,
            text=[f"{val:.3f}" for val in control_means],
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    # 添加实验组trace
    fig.add_trace(
        go.Bar(
            x=x_positions + 0.2,
            y=experimental_means,
            name="实验组",
            marker_color="red",
            error_y=dict(type="data", array=experimental_stds, visible=True),
            width=0.4,
            text=[f"{val:.3f}" for val in experimental_means],
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    # 设置x轴标签
    fig.update_xaxes(
        ticktext=[f"Block {i}" for i in range(3)], tickvals=x_positions, row=1, col=1
    )

    # 2. Rich命中率对比
    rich_hit_metric = "mean_rich_hit_rate"
    control_mean = np.mean(control_values[rich_hit_metric])
    control_std = np.std(control_values[rich_hit_metric], ddof=1)
    experimental_mean = np.mean(experimental_values[rich_hit_metric])
    experimental_std = np.std(experimental_values[rich_hit_metric], ddof=1)

    # 添加对照组
    fig.add_trace(
        go.Bar(
            x=["Rich命中率"],
            y=[control_mean],
            name="对照组",
            legendgroup="control",
            marker_color="green",
            error_y=dict(type="data", array=[control_std], visible=True),
            width=0.4,
            text=[f"{control_mean:.3f}"],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 添加实验组
    fig.add_trace(
        go.Bar(
            x=["Rich命中率"],
            y=[experimental_mean],
            name="实验组",
            legendgroup="experimental",
            marker_color="red",
            error_y=dict(type="data", array=[experimental_std], visible=True),
            width=0.4,
            text=[f"{experimental_mean:.3f}"],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Lean命中率对比
    lean_hit_metric = "mean_lean_hit_rate"
    control_mean = np.mean(control_values[lean_hit_metric])
    control_std = np.std(control_values[lean_hit_metric], ddof=1)
    experimental_mean = np.mean(experimental_values[lean_hit_metric])
    experimental_std = np.std(experimental_values[lean_hit_metric], ddof=1)

    # 添加对照组
    fig.add_trace(
        go.Bar(
            x=["Lean命中率"],
            y=[control_mean],
            name="对照组",
            legendgroup="control",
            marker_color="green",
            error_y=dict(type="data", array=[control_std], visible=True),
            width=0.4,
            text=[f"{control_mean:.3f}"],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # 添加实验组
    fig.add_trace(
        go.Bar(
            x=["Lean命中率"],
            y=[experimental_mean],
            name="实验组",
            legendgroup="experimental",
            marker_color="red",
            error_y=dict(type="data", array=[experimental_std], visible=True),
            width=0.4,
            text=[f"{experimental_mean:.3f}"],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    # 4. 条件性错误率对比（4个指标）
    probs_metrics = [
        "mean_lean_miss_after_rewarded_rich",
        "mean_lean_miss_after_nonrewarded_rich",
        "mean_rich_miss_after_rewarded_rich",
        "mean_rich_miss_after_rewarded_lean",
    ]
    probs_names = ["前富奖励贫", "前富不奖贫", "前富奖励富", "前贫奖励富"]

    # 准备数据
    x_positions_probs = np.arange(len(probs_metrics))
    control_means_probs = [np.mean(control_values[metric]) for metric in probs_metrics]
    control_stds_probs = [
        np.std(control_values[metric], ddof=1) for metric in probs_metrics
    ]
    experimental_means_probs = [
        np.mean(experimental_values[metric]) for metric in probs_metrics
    ]
    experimental_stds_probs = [
        np.std(experimental_values[metric], ddof=1) for metric in probs_metrics
    ]

    # 添加对照组
    fig.add_trace(
        go.Bar(
            x=x_positions_probs - 0.2,
            y=control_means_probs,
            name="对照组",
            legendgroup="control",
            marker_color="green",
            error_y=dict(type="data", array=control_stds_probs, visible=True),
            width=0.4,
            text=[f"{val:.3f}" for val in control_means_probs],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=4,
    )

    # 添加实验组
    fig.add_trace(
        go.Bar(
            x=x_positions_probs + 0.2,
            y=experimental_means_probs,
            name="实验组",
            legendgroup="experimental",
            marker_color="red",
            error_y=dict(type="data", array=experimental_stds_probs, visible=True),
            width=0.4,
            text=[f"{val:.3f}" for val in experimental_means_probs],
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=4,
    )

    # 设置x轴标签
    fig.update_xaxes(ticktext=probs_names, tickvals=x_positions_probs, row=1, col=4)

    # 5. 反应时差异对比
    rt_diff_metric = "mean_rt_diff"
    control_mean = np.mean(control_values[rt_diff_metric])
    control_std = np.std(control_values[rt_diff_metric], ddof=1)
    experimental_mean = np.mean(experimental_values[rt_diff_metric])
    experimental_std = np.std(experimental_values[rt_diff_metric], ddof=1)

    # 添加对照组
    fig.add_trace(
        go.Bar(
            x=["反应时差异"],
            y=[control_mean],
            name="对照组",
            legendgroup="control",
            marker_color="green",
            error_y=dict(type="data", array=[control_std], visible=True),
            width=0.4,
            text=[f"{control_mean:.3f}"],
            textposition="outside",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 添加实验组
    fig.add_trace(
        go.Bar(
            x=["反应时差异"],
            y=[experimental_mean],
            name="实验组",
            legendgroup="experimental",
            marker_color="red",
            error_y=dict(type="data", array=[experimental_std], visible=True),
            width=0.4,
            text=[f"{experimental_mean:.3f}"],
            textposition="outside",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 6. 反应偏向箱形图（3个block）
    # 为每个block创建箱形图
    for i, metric in enumerate(log_b_metrics):
        fig.add_trace(
            go.Box(
                y=control_values[metric],
                name=f"对照组-Block{i}",
                marker_color="lightgreen",
                boxmean="sd",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                showlegend=True if i == 0 else False,
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Box(
                y=experimental_values[metric],
                name=f"实验组-Block{i}",
                marker_color="lightcoral",
                boxmean="sd",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                showlegend=True if i == 0 else False,
            ),
            row=2,
            col=2,
        )

    # 7. 命中率箱形图
    # Rich命中率箱形图
    fig.add_trace(
        go.Box(
            y=control_values[rich_hit_metric],
            name="对照组-Rich命中率",
            marker_color="lightgreen",
            boxmean="sd",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    fig.add_trace(
        go.Box(
            y=experimental_values[rich_hit_metric],
            name="实验组-Rich命中率",
            marker_color="lightcoral",
            boxmean="sd",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    # Lean命中率箱形图
    fig.add_trace(
        go.Box(
            y=control_values[lean_hit_metric],
            name="对照组-Lean命中率",
            marker_color="darkgreen",
            boxmean="sd",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    fig.add_trace(
        go.Box(
            y=experimental_values[lean_hit_metric],
            name="实验组-Lean命中率",
            marker_color="darkred",
            boxmean="sd",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            showlegend=False,
        ),
        row=2,
        col=3,
    )

    # 8. 条件性错误率箱形图（新增）
    # 为每个条件性错误率指标创建箱形图
    for i, metric in enumerate(probs_metrics):
        fig.add_trace(
            go.Box(
                y=control_values[metric],
                name=f"对照组-{probs_names[i]}",
                marker_color="lightgreen",
                boxmean="sd",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                showlegend=True if i == 0 else False,
            ),
            row=2,
            col=4,
        )

        fig.add_trace(
            go.Box(
                y=experimental_values[metric],
                name=f"实验组-{probs_names[i]}",
                marker_color="lightcoral",
                boxmean="sd",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                showlegend=True if i == 0 else False,
            ),
            row=2,
            col=4,
        )

    fig.update_layout(
        title=dict(
            text="PRT组间比较分析报告",
            font=dict(size=22, family="Arial Black"),
            x=0.5,
        ),
        height=600 * 2,
        width=500 * 4,
        showlegend=True,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


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
    group_name: str = None,
    reference_group: Literal["control", "mdd"] = None,
):
    """组PRT分析"""
    if result_dir is None:
        result_dir = Path("prt_group_results")

    if group_name is not None:
        result_dir = result_dir / group_name

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
            result["key_metrics"]["subject_id"] = subject_id

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

    for metric in key_metrics:
        # 获取当前组的指标值
        group_values = [m[metric] for m in group_metrics]

        if len(group_values) < 2:
            statistical_results[metric] = {"error": "样本量不足"}
            continue

        if "log_b" in metric:
            ref_value = ref_values[metric]
        elif metric == "mean_hit_rate_diff":
            ref_value = ref_values["rich_hit_rate"] - ref_values["lean_hit_rate"]
        else:
            ref_value = ref_values[metric[5:]]

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
            "reference_value": float(ref_value),
            "t_statistic": float(t_stat),
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "effect_size_desc": effect_size_desc,
            "required_sample_size_per_group": sample_size_info.get("required_n")
            if sample_size_info
            else None,
            "required_total_sample_size": sample_size_info.get("required_n_total")
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
    # all_metrics_df.insert(0, "subject_id", [r["subject_id"] for r in all_results])
    all_metrics_df.to_csv(result_dir / "group_all_metrics.csv", index=False)

    group_mean_metrics = all_metrics_df.mean(numeric_only=True).to_dict()
    group_std_metrics = all_metrics_df.std(numeric_only=True).to_dict()

    stats_df = pd.DataFrame(
        [group_mean_metrics, group_std_metrics], index=["mean", "std"]
    ).T
    stats_df.to_csv(result_dir / "group_statistics.csv")

    sample_size_data = []
    for metric, result in statistical_results.items():
        if "required_sample_size_per_group" in result:
            sample_size_data.append(
                {
                    "metric": metric,
                    "effect_size": result.get("cohens_d"),
                    "required_per_group": result.get("required_sample_size_per_group"),
                    "required_total": result.get("required_total_sample_size"),
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

    fig_spec = create_single_group_visualizations(group_metrics)

    fig_common = create_common_single_group_figures(
        group_metrics, statistical_results, key_metrics, metric_names
    )

    figs = fig_spec + fig_common
    save_html_report(
        result_dir,
        f"prt-{group_name}_group-analysis_report",
        figs,
        title=f"PRT{group_name}组分析报告",
    )

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

    control_name = groups[0] if groups else "control"

    control_group_results = run_group_prt_analysis(
        control_files, result_dir, control_name, reference_group="control"
    )
    control_results = control_group_results["all_results"]
    control_metrics = control_group_results["group_metrics"]

    experimental_name = groups[1] if groups and len(groups) > 1 else "experimental"

    experimental_group_results = run_group_prt_analysis(
        experimental_files, result_dir, experimental_name, reference_group="mdd"
    )
    experimental_results = experimental_group_results["all_results"]
    experimental_metrics = experimental_group_results["group_metrics"]

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
    all_control_metrics_df.insert(0, "group", control_name)
    # all_control_metrics_df.insert(
    #     1, "subject_id", [r["subject_id"] for r in control_results]
    # )

    all_experimental_metrics_df = pd.DataFrame(
        [r["key_metrics"] for r in experimental_results]
    )
    all_experimental_metrics_df.insert(0, "group", experimental_name)
    # all_experimental_metrics_df.insert(
    #     1, "subject_id", [r["subject_id"] for r in experimental_results]
    # )

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

    control_stats.to_csv(result_dir / f"{control_name}_group_statistics.csv")
    experimental_stats.to_csv(result_dir / f"{experimental_name}_group_statistics.csv")

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
    fig_spec = create_multi_group_visualizations(control_metrics, experimental_metrics)

    fig_common = create_common_comparison_figures(
        comparison_results, key_metrics, metric_names
    )

    figs = [fig_spec] + fig_common
    save_html_report(
        save_dir=result_dir,
        save_name=f"prt-{control_name}_{experimental_name}_group-analysis_report",
        figures=figs,
        title=f"PRT{control_name}-{experimental_name}组间比较分析",
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


# [ ]: 这里可以共享, 几个 run 都可以


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
