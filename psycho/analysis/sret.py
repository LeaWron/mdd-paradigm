import json
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
        "endorsement_count": {"positive": 30, "negative": 2},
        "reaction_time": {"positive": 500, "negative": 1150},
    },
    "mdd": {
        "endorsement_count": {"positive": 12, "negative": 24},
        "reaction_time": {"positive": 900, "negative": 600},
    },
}

key_metrics = [
    "positive_bias",
    "rt_negative_minus_positive",
    "rt_endorsed_minus_not",
    "positive_endorsement_rate",
    "negative_endorsement_rate",
    "endorsement_rate",
]

metric_names = [
    "积极偏向(积极认可率-消极认可率)",
    "消极-积极RT差",
    "认同-不认同RT差",
    "积极认可率",
    "消极认可率",
    "总认可率",
]

# 新增：全局所有指标
all_metrics = [
    "positive_bias",
    "rt_negative_minus_positive",
    "rt_endorsed_minus_not",
    "positive_endorsement_rate",
    "negative_endorsement_rate",
    "endorsement_rate",
    "positive_endorsement_count",
    "negative_endorsement_count",
    "positive_rt",
    "negative_rt",
    "yes_rt",
    "no_rt",
    "positive_intensity",
    "negative_intensity",
    "intensity_negative_minus_positive",
]

all_metric_names = [
    "积极偏向(积极认可率-消极认可率)",
    "消极-积极RT差",
    "认同-不认同RT差",
    "积极认可率",
    "消极认可率",
    "总认可率",
    "积极词认同数",
    "消极词认同数",
    "积极词平均反应时",
    "消极词平均反应时",
    "认同平均反应时",
    "不认同平均反应时",
    "积极词平均符合程度",
    "消极词平均符合程度",
    "消极-积极符合程度差",
]

# 新增：词级指标（用于保存每个词的详细信息）
word_metrics = [
    "word_endorsement_count",
    "word_rejection_count",
    "word_endorsement_rate",
    "word_mean_intensity",
    "word_mean_rt",
]

word_metric_names = [
    "词认同次数",
    "词拒绝次数",
    "词认同率",
    "词平均符合程度",
    "词平均反应时",
]


def find_sret_files(data_dir: Path) -> list[Path]:
    """查找指定目录下的SRET实验结果文件"""
    EXP_TYPE = "sret"
    return find_exp_files(data_dir, EXP_TYPE)


def load_and_preprocess_data(df: pl.DataFrame) -> pl.DataFrame:
    """加载并预处理数据"""
    try:
        trials_df = extract_trials_by_block(
            df,
            target_block_indices=["Encoding"],
            block_col="phase",
            trial_col="trial_index",
            fill_na=True,
        )

        if trials_df.height == 0:
            print("❌ 错误: 未找到有效的试次数据")
            return None

        required_columns = ["stim_word", "response", "rt", "stim_type"]
        missing_columns = [
            col for col in required_columns if col not in trials_df.columns
        ]
        if missing_columns:
            print(f"❌ 缺少必要列: {missing_columns}")
            return None

        trials_df = trials_df.with_columns(
            (
                pl.when(pl.col("rt").is_null())
                .then(pl.col("rt").max() + 1)
                .otherwise(pl.col("rt"))
            ),
        )
        trials_df = trials_df.with_columns(
            (pl.col("rt") * 1000).alias("rt_ms"),
        )

        if "intensity" in trials_df.columns:
            trials_df = trials_df.with_columns(
                pl.col("intensity").clip(0, 10).alias("intensity")
            )

        trials_df = trials_df.with_columns(
            (pl.col("response") == "yes").cast(pl.Int8).alias("response_code")
        )

        return trials_df

    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
        return None


def calculate_word_level_metrics(df: pl.DataFrame) -> dict[str, Any]:
    """计算词级指标：每个词的认同次数、拒绝次数、符合程度和反应时"""
    word_results = {}

    # 按词分组统计
    word_stats = df.group_by(["stim_word", "stim_type"]).agg(
        [
            pl.col("response")
            .filter(pl.col("response") == "yes")
            .count()
            .alias("endorsement_count"),
            pl.col("response")
            .filter(pl.col("response") == "no")
            .count()
            .alias("rejection_count"),
            pl.col("intensity").mean().alias("mean_intensity")
            if "intensity" in df.columns
            else pl.lit(None).alias("mean_intensity"),
            pl.col("rt_ms").mean().alias("mean_rt"),
            pl.count().alias("total_trials"),
        ]
    )

    # 计算词认同率
    word_stats = word_stats.with_columns(
        (pl.col("endorsement_count") / pl.col("total_trials")).alias("endorsement_rate")
    )

    # 转换为字典格式
    word_results["word_stats"] = word_stats.to_dicts()

    # 按词性汇总
    valence_stats = word_stats.group_by("stim_type").agg(
        [
            pl.col("endorsement_count").sum().alias("total_endorsement_count"),
            pl.col("rejection_count").sum().alias("total_rejection_count"),
            pl.col("mean_intensity").mean().alias("mean_intensity")
            if "intensity" in df.columns
            else pl.lit(None).alias("mean_intensity"),
            pl.col("mean_rt").mean().alias("mean_rt"),
            pl.col("endorsement_rate").mean().alias("mean_endorsement_rate"),
        ]
    )

    word_results["valence_stats"] = valence_stats.to_dicts()

    return word_results


def calculate_all_metrics_single(df: pl.DataFrame) -> dict[str, float]:
    """计算单个被试的所有指标（包括词级指标）"""
    metrics = calculate_key_metrics_single(df)

    # 计算词级指标（汇总）
    word_results = calculate_word_level_metrics(df)

    # 添加词级汇总指标到metrics
    for valence_stat in word_results.get("valence_stats", []):
        valence = valence_stat["stim_type"]
        metrics[f"{valence}_endorsement_count"] = valence_stat.get(
            "total_endorsement_count", 0
        )
        metrics[f"{valence}_rejection_count"] = valence_stat.get(
            "total_rejection_count", 0
        )

        if valence_stat.get("mean_intensity") is not None:
            metrics[f"{valence}_intensity"] = valence_stat.get("mean_intensity")

        if valence_stat.get("mean_rt") is not None:
            metrics[f"{valence}_rt"] = valence_stat.get("mean_rt")

    return metrics


def calculate_key_metrics_single(df: pl.DataFrame) -> dict[str, float]:
    """计算单个被试的关键指标"""
    metrics = {}

    total = df.height
    yes_count = df.filter(pl.col("response") == "yes").height
    yes_pos_count = df.filter(
        (pl.col("response") == "yes") & (pl.col("stim_type") == "positive")
    ).height
    yes_neg_count = df.filter(
        (pl.col("response") == "yes") & (pl.col("stim_type") == "negative")
    ).height

    endorsement_rate = yes_count / total if total > 0 else 0
    positive_rate = yes_pos_count / total if total > 0 else 0
    negative_rate = yes_neg_count / total if total > 0 else 0

    metrics["endorsement_rate"] = endorsement_rate
    metrics["positive_endorsement_rate"] = positive_rate
    metrics["negative_endorsement_rate"] = negative_rate

    # 按词性分组统计
    valence_stats = df.group_by("stim_type").agg(
        [
            pl.col("response")
            .filter(pl.col("response") == "yes")
            .count()
            .alias("yes_count"),
            pl.col("response").count().alias("total_count"),
            pl.col("rt_ms").mean().alias("mean_rt"),
        ]
    )

    valence_stats = valence_stats.with_columns(
        [(pl.col("yes_count") / pl.col("total_count")).alias("endorsement_rate")]
    )

    valence_dict = {}
    for row in valence_stats.to_dicts():
        valence_dict[row["stim_type"]] = row

    # 2. 积极偏向: 积极认同率 - 消极认同率
    if "positive" in valence_dict and "negative" in valence_dict:
        positive_rate = valence_dict["positive"]["endorsement_rate"]
        negative_rate = valence_dict["negative"]["endorsement_rate"]
        metrics["positive_bias"] = positive_rate - negative_rate
        metrics["positive_endorsement_rate"] = positive_rate
        metrics["negative_endorsement_rate"] = negative_rate
        metrics["positive_endorsement_count"] = valence_dict["positive"]["yes_count"]
        metrics["negative_endorsement_count"] = valence_dict["negative"]["yes_count"]

    # 3. 消极RT - 积极RT
    if "positive" in valence_dict and "negative" in valence_dict:
        negative_rt = valence_dict["negative"]["mean_rt"]
        positive_rt = valence_dict["positive"]["mean_rt"]
        metrics["rt_negative_minus_positive"] = negative_rt - positive_rt
        metrics["positive_rt"] = positive_rt
        metrics["negative_rt"] = negative_rt

    # 4. 认同RT - 不认同RT
    yes_rt = df.filter(pl.col("response") == "yes")["rt_ms"].mean()
    no_rt = df.filter(pl.col("response") == "no")["rt_ms"].mean()
    metrics["rt_endorsed_minus_not"] = yes_rt - no_rt
    metrics["yes_rt"] = yes_rt
    metrics["no_rt"] = no_rt

    # 5. 消极intensity - 积极intensity
    if (
        "intensity" in df.columns
        and "positive" in valence_dict
        and "negative" in valence_dict
    ):
        intensity_by_valence = df.group_by("stim_type").agg(
            [
                pl.col("intensity").mean().alias("mean_intensity"),
            ]
        )

        intensity_dict = {}
        for row in intensity_by_valence.to_dicts():
            intensity_dict[row["stim_type"]] = row

        if "positive" in intensity_dict and "negative" in intensity_dict:
            negative_intensity = intensity_dict["negative"]["mean_intensity"]
            positive_intensity = intensity_dict["positive"]["mean_intensity"]
            metrics["intensity_negative_minus_positive"] = (
                negative_intensity - positive_intensity
            )
            metrics["positive_intensity"] = positive_intensity
            metrics["negative_intensity"] = negative_intensity

    return metrics


def analyze_valence_performance(df: pl.DataFrame) -> dict[str, Any]:
    """分析词性表现"""
    results = {}

    valence_stats = df.group_by("stim_type").agg(
        [
            pl.col("response")
            .filter(pl.col("response") == "yes")
            .count()
            .alias("yes_count"),
            pl.col("response").count().alias("total_count"),
            pl.col("rt_ms").mean().alias("mean_rt"),
            pl.col("rt_ms").std().alias("std_rt"),
            pl.col("rt_ms").median().alias("median_rt"),
        ]
    )

    valence_stats = valence_stats.with_columns(
        [(pl.col("yes_count") / pl.col("total_count")).alias("endorsement_rate")]
    )

    results["valence_stats"] = valence_stats.to_dicts()

    if "intensity" in df.columns:
        intensity_by_valence = df.group_by("stim_type").agg(
            [
                pl.col("intensity").mean().alias("mean_intensity"),
                pl.col("intensity").std().alias("std_intensity"),
                pl.col("intensity").median().alias("median_intensity"),
            ]
        )
        results["intensity_stats"] = intensity_by_valence.to_dicts()

    return results


def analyze_reaction_time_breakdown(df: pl.DataFrame) -> dict[str, Any]:
    """更精细分析反应时"""
    results = {}

    # 按反应类型和词性分析
    response_valence_stats = df.group_by(["response", "stim_type"]).agg(
        [
            pl.col("rt_ms").mean().alias("mean_rt"),
            pl.col("rt_ms").std().alias("std_rt"),
            pl.count().alias("count"),
        ]
    )

    results["response_valence_stats"] = response_valence_stats.to_dicts()

    return results


def create_word_level_visualizations_single(
    word_stats: list[dict], result_dir: Path = None
) -> go.Figure:
    """创建单人分析的词级可视化"""
    if not word_stats:
        return None

    # 按词性排序：先积极词后消极词
    positive_words = [w for w in word_stats if w["stim_type"] == "positive"]
    negative_words = [w for w in word_stats if w["stim_type"] == "negative"]

    # 按认同率排序
    positive_words.sort(key=lambda x: x["endorsement_rate"], reverse=True)
    negative_words.sort(key=lambda x: x["endorsement_rate"], reverse=True)

    sorted_words = positive_words + negative_words

    words = [w["stim_word"] for w in sorted_words]
    stim_types = [w["stim_type"] for w in sorted_words]
    endorsement_rates = [w["endorsement_rate"] for w in sorted_words]
    mean_rts = [w.get("mean_rt", 0) for w in sorted_words]

    # 创建子图
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "每个词的平均反应时",
            "每个词的认同率",
            "词性分布统计",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15,
    )

    # 颜色映射
    colors = ["lightblue" if t == "positive" else "lightcoral" for t in stim_types]

    # 图1:每个词的平均反应时
    fig.add_trace(
        go.Bar(
            x=words,
            y=mean_rts,
            name="平均反应时(ms)",
            marker_color=colors,
            text=[f"{rt:.0f}ms" for rt in mean_rts],
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    # 图2：每个词的认同率
    fig.add_trace(
        go.Scatter(
            x=words,
            y=endorsement_rates,
            mode="lines+markers",
            name="认同率",
            line=dict(color="blue", width=2),
            marker=dict(
                size=8, color=colors, line=dict(width=1, color="DarkSlateGrey")
            ),
            text=[f"{rate:.1%}" for rate in endorsement_rates],
            textposition="top center",
        ),
        row=2,
        col=1,
    )

    # 图3：词性分布统计
    positive_endorsed = sum([w["endorsement_count"] for w in positive_words])
    positive_rejected = sum([w["rejection_count"] for w in positive_words])
    negative_endorsed = sum([w["endorsement_count"] for w in negative_words])
    negative_rejected = sum([w["rejection_count"] for w in negative_words])

    fig.add_trace(
        go.Bar(
            x=["积极词", "消极词"],
            y=[positive_endorsed, negative_endorsed],
            name="认同",
            marker_color=["lightblue", "lightcoral"],
            text=[positive_endorsed, negative_endorsed],
            textposition="auto",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=["积极词", "消极词"],
            y=[positive_rejected, negative_rejected],
            name="拒绝",
            marker_color=["powderblue", "mistyrose"],
            text=[positive_rejected, negative_rejected],
            textposition="auto",
        ),
        row=3,
        col=1,
    )

    # 更新布局
    fig.update_layout(
        title=dict(
            text="SRET词级分析报告", font=dict(size=20, family="Arial Black"), x=0.5
        ),
        height=400 * 3,
        width=1400,
        showlegend=True,
        template="plotly_white",
        barmode="stack",  # 对于堆叠条形图
    )

    # 更新x轴标签角度
    fig.update_xaxes(tickangle=90, row=1, col=1)
    fig.update_xaxes(tickangle=90, row=2, col=1)
    fig.update_xaxes(tickangle=90, row=3, col=1)

    if result_dir:
        fig.write_html(str(result_dir / "sret_word_level_analysis.html"))

    return fig


def create_single_visualizations(
    metrics: dict[str, float],
    valence_results: dict[str, Any],
    result_dir: Path,
    word_results: dict[str, Any] = None,
) -> list[go.Figure]:
    """创建单人分析可视化"""
    figs = []

    # 原有的可视化图
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "1. 认同数量对比",
            "2. 反应时对比",
            "3. 认同率对比",
            "4. 关键指标",
            "5. 与参考值对比",
            "6. 反应时分布",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "table"}, {"type": "scatter"}, {"type": "histogram"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.15,
    )

    # 图1: 认同数量对比
    if (
        "positive_endorsement_count" in metrics
        and "negative_endorsement_count" in metrics
    ):
        fig.add_trace(
            go.Bar(
                x=["积极词", "消极词"],
                y=[
                    metrics["positive_endorsement_count"],
                    metrics["negative_endorsement_count"],
                ],
                name="当前被试",
                marker_color=["lightblue", "lightcoral"],
                text=[
                    str(int(metrics["positive_endorsement_count"])),
                    str(int(metrics["negative_endorsement_count"])),
                ],
                textposition="auto",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=["积极词", "消极词"],
                y=[
                    REFERENCE_VALUES["control"]["endorsement_count"]["positive"],
                    REFERENCE_VALUES["control"]["endorsement_count"]["negative"],
                ],
                mode="markers",
                name="对照组参考",
                marker=dict(size=12, color="green", symbol="diamond"),
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=["积极词", "消极词"],
                y=[
                    REFERENCE_VALUES["mdd"]["endorsement_count"]["positive"],
                    REFERENCE_VALUES["mdd"]["endorsement_count"]["negative"],
                ],
                mode="markers",
                name="mdd参考",
                marker=dict(size=12, color="red", symbol="diamond"),
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

    # 图2: 反应时对比
    if "positive_rt" in metrics and "negative_rt" in metrics:
        fig.add_trace(
            go.Bar(
                x=["积极词", "消极词"],
                y=[metrics["positive_rt"], metrics["negative_rt"]],
                name="当前被试",
                marker_color=["lightblue", "lightcoral"],
                text=[
                    f"{metrics['positive_rt']:.0f} ms",
                    f"{metrics['negative_rt']:.0f} ms",
                ],
                textposition="auto",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=["积极词", "消极词"],
                y=[
                    REFERENCE_VALUES["control"]["reaction_time"]["positive"],
                    REFERENCE_VALUES["control"]["reaction_time"]["negative"],
                ],
                mode="markers",
                name="对照组参考",
                marker=dict(size=12, color="green", symbol="diamond"),
                opacity=0.7,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=["积极词", "消极词"],
                y=[
                    REFERENCE_VALUES["mdd"]["reaction_time"]["positive"],
                    REFERENCE_VALUES["mdd"]["reaction_time"]["negative"],
                ],
                mode="markers",
                name="mdd参考",
                marker=dict(size=12, color="red", symbol="diamond"),
                opacity=0.7,
            ),
            row=1,
            col=2,
        )

    # 图3: 认同率对比
    if (
        "positive_endorsement_rate" in metrics
        and "negative_endorsement_rate" in metrics
    ):
        fig.add_trace(
            go.Bar(
                x=["积极词", "消极词"],
                y=[
                    metrics["positive_endorsement_rate"],
                    metrics["negative_endorsement_rate"],
                ],
                name="当前被试",
                marker_color=["lightblue", "lightcoral"],
                text=[
                    f"{metrics['positive_endorsement_rate']:.1%}",
                    f"{metrics['negative_endorsement_rate']:.1%}",
                ],
                textposition="auto",
            ),
            row=1,
            col=3,
        )

    # 图4: 关键指标表格
    key_metrics_display = [
        ["指标", "值"],
        ["总认同率", f"{metrics.get('endorsement_rate', 0):.1%}"],
        ["积极偏向", f"{metrics.get('positive_bias', 0):.3f}"],
        ["消极RT - 积极RT", f"{metrics.get('rt_negative_minus_positive', 0):.1f} ms"],
        ["认同RT - 不认同RT", f"{metrics.get('rt_endorsed_minus_not', 0):.1f} ms"],
    ]

    if "intensity_negative_minus_positive" in metrics:
        key_metrics_display.append(
            [
                "消极强度 - 积极强度",
                f"{metrics['intensity_negative_minus_positive']:.2f}",
            ]
        )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["指标", "值"],
                fill_color="lightblue",
                align="left",
                font=dict(size=12),
            ),
            cells=dict(
                values=[
                    [row[0] for row in key_metrics_display[1:]],
                    [row[1] for row in key_metrics_display[1:]],
                ],
                fill_color="lavender",
                align="left",
                font=dict(size=11),
            ),
        ),
        row=2,
        col=1,
    )

    # 图5: 积极偏向对比（参考）
    if "positive_bias" in metrics:
        hco_positive_rate = (
            REFERENCE_VALUES["control"]["endorsement_count"]["positive"] / 40
        )
        hco_negative_rate = (
            REFERENCE_VALUES["control"]["endorsement_count"]["negative"] / 40
        )
        hco_bias = hco_positive_rate - hco_negative_rate

        mdd_positive_rate = (
            REFERENCE_VALUES["mdd"]["endorsement_count"]["positive"] / 40
        )
        mdd_negative_rate = (
            REFERENCE_VALUES["mdd"]["endorsement_count"]["negative"] / 40
        )
        mdd_bias = mdd_positive_rate - mdd_negative_rate

        fig.add_trace(
            go.Bar(
                x=["当前被试", "对照组参考", "mdd参考"],
                y=[metrics["positive_bias"], hco_bias, mdd_bias],
                name="积极偏向",
                marker_color=["blue", "green", "red"],
                text=[
                    f"{metrics['positive_bias']:.3f}",
                    f"{hco_bias:.3f}",
                    f"{mdd_bias:.3f}",
                ],
                textposition="auto",
            ),
            row=2,
            col=2,
        )

    # 图6: 反应时分布（简化版）
    if "valence_stats" in valence_results:
        # 准备数据
        rt_data = []
        labels = []
        for stat in valence_results["valence_stats"]:
            rt_data.append(stat["mean_rt"])
            labels.append(f"{stat['stim_type']}\n({stat['mean_rt']:.0f}ms)")

        fig.add_trace(
            go.Bar(
                x=labels,
                y=rt_data,
                name="平均反应时",
                marker_color=["lightblue", "lightcoral"],
            ),
            row=2,
            col=3,
        )

    # 更新布局
    fig.update_layout(
        title=dict(
            text="SRET分析报告", font=dict(size=24, family="Arial Black"), x=0.5
        ),
        height=900,
        width=1400,
        showlegend=True,
        template="plotly_white",
    )

    fig.write_html(str(result_dir / "sret_analysis_report.html"))
    figs.append(fig)
    if word_results and "word_stats" in word_results:
        word_fig = create_word_level_visualizations_single(
            word_results["word_stats"], result_dir
        )
        if word_fig:
            figs.append(word_fig)
    return figs


def save_results(
    metrics: dict[str, float],
    valence_results: dict[str, Any],
    result_dir: Path,
    word_results: dict[str, Any] = None,
):
    """保存结果"""
    metrics_df = pl.DataFrame([metrics])
    metrics_df.write_csv(result_dir / "sret_key_metrics.csv")

    if "valence_stats" in valence_results:
        valence_df = pl.DataFrame(valence_results["valence_stats"])
        valence_df.write_csv(result_dir / "sret_valence_stats.csv")

    # 保存词级结果
    if word_results and "word_stats" in word_results:
        word_df = pl.DataFrame(word_results["word_stats"])
        word_df.write_csv(result_dir / "sret_word_level_stats.csv")

    results = {
        "key_metrics": metrics,
        "valence_analysis": valence_results,
    }

    if word_results:
        results["word_level_analysis"] = word_results

    with open(result_dir / "sret_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 分析完成! 结果保存在: {result_dir}")


def analyze_sret_data_single(
    df: pl.DataFrame,
    target_blocks: list[str] = ["Encoding"],
    result_dir: Path = Path("results"),
) -> dict[str, Any]:
    """分析单个被试的SRET数据"""

    # 1. 提取试次数据
    trials_df = load_and_preprocess_data(df)

    # 2. 计算所有指标（包括词级指标）
    metrics = calculate_all_metrics_single(trials_df)

    # 3. 词性表现分析
    valence_results = analyze_valence_performance(trials_df)

    # 4. 词级指标分析
    word_results = calculate_word_level_metrics(trials_df)

    # 5. 反应时细分分析
    rt_breakdown = analyze_reaction_time_breakdown(trials_df)

    # 6. 创建可视化
    figs = create_single_visualizations(  # noqa: F841
        metrics, valence_results, result_dir, word_results
    )

    # 7. 保存结果
    save_results(metrics, valence_results, result_dir, word_results)

    # 8. 生成报告
    results = {
        "data_summary": {
            "total_trials": trials_df.height,
            "positive_words": trials_df.filter(
                pl.col("stim_type") == "positive"
            ).height,
            "negative_words": trials_df.filter(
                pl.col("stim_type") == "negative"
            ).height,
            "endorsed_trials": trials_df.filter(pl.col("response") == "yes").height,
            "not_endorsed_trials": trials_df.filter(pl.col("response") == "no").height,
        },
        "key_metrics": metrics,
        "valence_results": valence_results,
        "word_results": word_results,
        "rt_breakdown": rt_breakdown,
    }

    return results


def create_word_level_visualizations_group(
    all_word_stats: list[list[dict]],  # 每个被试的词级数据列表
    result_dir: Path = None,
) -> list[go.Figure]:
    """创建单组分析的词级可视化 - 横向条形图"""
    if not all_word_stats:
        return []

    figs = []

    # 收集所有被试的数据
    word_data = {}
    for subject_word_stats in all_word_stats:
        for word_stat in subject_word_stats:
            word = word_stat["stim_word"]
            stim_type = word_stat["stim_type"]

            if word not in word_data:
                word_data[word] = {
                    "stim_type": stim_type,
                    "endorsement_counts": [],
                    "rejection_counts": [],
                    "mean_rts": [],
                }

            word_data[word]["endorsement_counts"].append(word_stat["endorsement_count"])
            word_data[word]["rejection_counts"].append(word_stat["rejection_count"])

            if word_stat.get("mean_rt") is not None:
                word_data[word]["mean_rts"].append(word_stat["mean_rt"])

    # 计算每个词的组统计
    words = []
    stim_types = []
    mean_endorsement_counts = []
    std_endorsement_counts = []
    mean_rejection_counts = []
    std_rejection_counts = []
    endorsement_rates = []
    mean_rts = []
    std_rts = []
    se_rts = []  # 标准误

    for word, data in word_data.items():
        words.append(word)
        stim_types.append(data["stim_type"])

        # 平均选择次数和标准差
        mean_endorsement_counts.append(np.mean(data["endorsement_counts"]))
        std_endorsement_counts.append(
            np.std(data["endorsement_counts"], ddof=1)
            if len(data["endorsement_counts"]) > 1
            else 0
        )
        mean_rejection_counts.append(np.mean(data["rejection_counts"]))
        std_rejection_counts.append(
            np.std(data["rejection_counts"], ddof=1)
            if len(data["rejection_counts"]) > 1
            else 0
        )

        # 平均认同率
        total_trials = np.sum(data["endorsement_counts"]) + np.sum(
            data["rejection_counts"]
        )
        if total_trials > 0:
            endorsement_rate = np.sum(data["endorsement_counts"]) / total_trials
        else:
            endorsement_rate = 0
        endorsement_rates.append(endorsement_rate)

        # 平均反应时 - 确保有有效数据
        if data["mean_rts"] and not all(np.isnan(rt) for rt in data["mean_rts"]):
            valid_rts = [rt for rt in data["mean_rts"] if not np.isnan(rt)]
            if valid_rts:
                mean_rt = np.mean(valid_rts)
                std_rt = np.std(valid_rts, ddof=1) if len(valid_rts) > 1 else 0
                se_rt = std_rt / np.sqrt(len(valid_rts)) if len(valid_rts) > 0 else 0
                mean_rts.append(mean_rt)
                std_rts.append(std_rt)
                se_rts.append(se_rt)
            else:
                mean_rts.append(0)
                std_rts.append(0)
                se_rts.append(0)
        else:
            mean_rts.append(0)
            std_rts.append(0)
            se_rts.append(0)

    # 按词性排序
    positive_indices = [i for i, t in enumerate(stim_types) if t == "positive"]
    negative_indices = [i for i, t in enumerate(stim_types) if t == "negative"]

    sorted_indices = positive_indices + negative_indices
    words = [words[i] for i in sorted_indices]
    stim_types = [stim_types[i] for i in sorted_indices]
    mean_endorsement_counts = [mean_endorsement_counts[i] for i in sorted_indices]
    std_endorsement_counts = [std_endorsement_counts[i] for i in sorted_indices]
    mean_rejection_counts = [mean_rejection_counts[i] for i in sorted_indices]
    std_rejection_counts = [std_rejection_counts[i] for i in sorted_indices]
    endorsement_rates = [endorsement_rates[i] for i in sorted_indices]
    mean_rts = [mean_rts[i] for i in sorted_indices]
    std_rts = [std_rts[i] for i in sorted_indices]
    se_rts = [se_rts[i] for i in sorted_indices]

    # 1. 创建选择次数图
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "选择次数",
            "反应时间对比",
            "认同率",
        ),
        specs=[[{"type": "bar"}], [{"type": "bar"}], [{"type": "bar"}]],
    )

    # 只显示前30个词，避免图表过于拥挤
    display_limit = min(80, len(words))
    display_words = words[:display_limit]
    display_stim_types = stim_types[:display_limit]
    display_mean_endorsed = mean_endorsement_counts[:display_limit]
    display_std_endorsed = std_endorsement_counts[:display_limit]
    display_mean_rejected = mean_rejection_counts[:display_limit]
    display_std_rejected = std_rejection_counts[:display_limit]

    # 积极词和消极词分开显示
    for i, (
        word,
        stim_type,
        mean_endorsed,
        std_endorsed,
        mean_rejected,
        std_rejected,
    ) in enumerate(
        zip(
            display_words,
            display_stim_types,
            display_mean_endorsed,
            display_std_endorsed,
            display_mean_rejected,
            display_std_rejected,
        )
    ):
        fig.add_trace(
            go.Bar(
                x=[word],
                y=[mean_endorsed],
                name=f"{word} (认同)",
                marker_color="lightblue" if stim_type == "positive" else "lightcoral",
                text=[f"{mean_endorsed:.2f}"],
                textposition="auto",
                showlegend=False,
                hovertemplate=f"词: {word}<br>词性: {stim_type}<br>平均认同次数: {mean_endorsed:.2f}<br>标准差: {std_endorsed:.2f}<br><extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=[word],
                y=[mean_rejected],
                name=f"{word} (拒绝)",
                marker_color="powderblue" if stim_type == "positive" else "mistyrose",
                text=[f"{mean_rejected:.2f}"],
                textposition="auto",
                showlegend=False,
                hovertemplate=f"词: {word}<br>词性: {stim_type}<br>平均拒绝次数: {mean_rejected:.2f}<br>标准差: {std_rejected:.2f}<br><extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 更新x轴标签角度
        fig.update_xaxes(tickangle=45, row=1, col=1)

    # 2. 创建反应时图 - 确保有有效数据
    if any(rt > 0 for rt in mean_rts):
        # 只显示前30个词
        display_mean_rts = mean_rts[:display_limit]
        display_std_rts = std_rts[:display_limit]
        display_se_rts = se_rts[:display_limit]

        for i, (word, stim_type, mean_rt, std_rt, se_rt) in enumerate(
            zip(
                display_words,
                display_stim_types,
                display_mean_rts,
                display_std_rts,
                display_se_rts,
            )
        ):
            if mean_rt > 0:  # 只显示有有效数据的词
                fig.add_trace(
                    go.Bar(
                        x=[word],
                        y=[mean_rt],
                        name=word,
                        marker_color="lightblue"
                        if stim_type == "positive"
                        else "lightcoral",
                        text=[f"{mean_rt:.0f}±{std_rt:.0f}ms"],
                        textposition="auto",
                        showlegend=False,
                        error_y=dict(
                            type="data",
                            array=[std_rt],
                            visible=True,
                            color="rgba(0,0,0,0.5)",
                        ),
                        hovertemplate=f"词: {word}<br>词性: {stim_type}<br>平均反应时: {mean_rt:.0f}ms<br>标准差: {std_rt:.0f}ms<br>标准误: {se_rt:.0f}ms<br><extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

                # 更新x轴标签角度
                fig.update_xaxes(tickangle=45, row=2, col=1)

    # 3. 创建认同率图 - 计算认同率的标准误

    # 计算认同率的标准误
    endorsement_rate_ses = []
    for i, (rate, mean_endorsed, mean_rejected) in enumerate(
        zip(endorsement_rates, mean_endorsement_counts, mean_rejection_counts)
    ):
        n = len(all_word_stats)  # 被试数
        if n > 0 and rate > 0 and rate < 1:
            # 使用二项分布标准误公式
            se = np.sqrt(rate * (1 - rate) / n) * 100  # 转换为百分比
        else:
            se = 0
        endorsement_rate_ses.append(se)

    # 只显示前30个词
    display_endorsement_rates = endorsement_rates[:display_limit]
    display_endorsement_rate_ses = endorsement_rate_ses[:display_limit]

    for i, (word, stim_type, rate, se) in enumerate(
        zip(
            display_words,
            display_stim_types,
            display_endorsement_rates,
            display_endorsement_rate_ses,
        )
    ):
        fig.add_trace(
            go.Bar(
                x=[word],
                y=[rate * 100],  # 转换为百分比
                name=word,
                marker_color="lightblue" if stim_type == "positive" else "lightcoral",
                text=[f"{rate:.1%}"],
                textposition="auto",
                showlegend=False,
                error_y=dict(
                    type="data",
                    array=[se],
                    visible=True,
                    color="rgba(0,0,0,0.5)",
                ),
                hovertemplate=f"词: {word}<br>词性: {stim_type}<br>平均认同率: {rate:.1%}<br>标准误: {se:.1f}%<br><extra></extra>",
            ),
            row=3,
            col=1,
        )

    # 更新x轴标签角度
    fig.update_xaxes(tickangle=45, row=3, col=1)

    if result_dir:
        # 保存完整的词级数据到CSV
        word_stats_df = pd.DataFrame(
            {
                "word": words,
                "stim_type": stim_types,
                "mean_endorsement_count": mean_endorsement_counts,
                "std_endorsement_count": std_endorsement_counts,
                "mean_rejection_count": mean_rejection_counts,
                "std_rejection_count": std_rejection_counts,
                "endorsement_rate": endorsement_rates,
                "endorsement_rate_se(%)": [se for se in endorsement_rate_ses],
                "mean_rt": mean_rts,
                "std_rt": std_rts,
                "se_rt": se_rts,
            }
        )
        word_stats_df.to_csv(
            result_dir / "sret_group_word_level_stats.csv", index=False
        )
    # 积极消极分界线
    fig.add_vline(
        x=len(positive_indices) - 0.5,
        line_width=3,
        line_dash="dash",
        line_color="gray",
    )

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"SRET组分析词级报告 - (显示前{display_limit}个词)",
            font=dict(size=16, family="Arial Black"),
            x=0.5,
        ),
        width=2000,
        height=600 * 3,
        showlegend=True,
        template="plotly_white",
        hovermode="x unified",
    )
    figs.append(fig)
    return figs


def create_single_group_visualizations(
    group_metrics: list[dict[str, float]],
    all_results: list[dict] | None = None,
) -> list[go.Figure]:
    """单个组的组分析可视化"""
    all_metrics = group_metrics
    figs = []

    # 原有的可视化图
    fig_1 = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "1. 各被试积极偏向",
            "2. 各被试消极积极反应差",
            "3. 各被试认同不认同反应差",
            "4. 积极偏向分布",
            "5. 消极积极反应差分布",
            "6. 认同不认同反应差分布",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "box"}, {"type": "box"}, {"type": "box"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    subject_ids = [(int(m["subject_id"]), i) for i, m in enumerate(all_metrics)]
    subject_ids.sort()
    all_metrics = [all_metrics[m[1]] for m in subject_ids]

    subjects = [f"s{m[0]}" for m in subject_ids]

    # 图1: 各被试积极偏向分布
    bias_values = [m["positive_bias"] for m in all_metrics]

    fig_1.add_trace(
        go.Bar(
            x=subjects,
            y=bias_values,
            name="积极偏向",
            marker_color="lightblue",
            text=[f"{v:.3f}" for v in bias_values],
            textposition="auto",
        ),
        row=1,
        col=1,
    )
    fig_1.add_trace(
        go.Box(
            y=bias_values,
            boxmean="sd",
            name="积极偏向",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        ),
        row=2,
        col=1,
    )

    # 图2: 各被试消极积极反应差分布
    diff_values = [m["rt_negative_minus_positive"] for m in all_metrics]

    fig_1.add_trace(
        go.Bar(
            x=subjects,
            y=diff_values,
            name="消极-积极RT差",
            marker_color="salmon",
            text=[f"{v:.3f}" for v in diff_values],
            textposition="auto",
        ),
        row=1,
        col=2,
    )
    fig_1.add_trace(
        go.Box(
            y=diff_values,
            boxmean="sd",
            name="消极-积极RT差",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        ),
        row=2,
        col=2,
    )

    # 图3: 各被试认同不认同反应差分布
    diff_values = [m["rt_endorsed_minus_not"] for m in all_metrics]

    fig_1.add_trace(
        go.Bar(
            x=subjects,
            y=diff_values,
            name="认同-不认同RT差",
            marker_color="mediumseagreen",
            text=[f"{v:.3f}" for v in diff_values],
            textposition="auto",
        ),
        row=1,
        col=3,
    )
    fig_1.add_trace(
        go.Box(
            y=diff_values,
            boxmean="sd",
            name="认同-不认同RT差",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        ),
        row=2,
        col=3,
    )

    # 更新布局
    fig_1.update_layout(
        height=400 * 2,
        width=1600,
        showlegend=True,
        template="plotly_white",
    )
    figs.append(fig_1)

    fig_2 = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "1. 各被试积极认可率",
            "2. 各被试消极认可率",
            "3. 各被试总认可率",
            "4. 积极认可率分布",
            "5. 消极认可率分布",
            "6. 总认可率分布",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "box"}, {"type": "box"}, {"type": "box"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # 图1: 各被试积极认可率分布
    positive_values = [m["positive_endorsement_rate"] for m in all_metrics]
    fig_2.add_trace(
        go.Bar(
            x=subjects,
            y=positive_values,
            name="积极认可率",
            marker_color="lightblue",
            text=[f"{v:.3f}" for v in positive_values],
            textposition="auto",
        ),
        row=1,
        col=1,
    )
    fig_2.add_trace(
        go.Box(
            y=positive_values,
            boxmean="sd",
            name="积极认可率",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        ),
        row=2,
        col=1,
    )

    # 图2: 各被试消极认可率分布
    negative_values = [m["negative_endorsement_rate"] for m in all_metrics]
    fig_2.add_trace(
        go.Bar(
            x=subjects,
            y=negative_values,
            name="消极认可率",
            marker_color="salmon",
            text=[f"{v:.3f}" for v in negative_values],
            textposition="auto",
        ),
        row=1,
        col=2,
    )
    fig_2.add_trace(
        go.Box(
            y=negative_values,
            boxmean="sd",
            name="消极认可率",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        ),
        row=2,
        col=2,
    )
    # 图3: 各被试总认可率分布
    total_values = [m["endorsement_rate"] for m in all_metrics]
    fig_2.add_trace(
        go.Bar(
            x=subjects,
            y=total_values,
            name="总认可率",
            marker_color="mediumseagreen",
            text=[f"{v:.3f}" for v in total_values],
            textposition="auto",
        ),
        row=1,
        col=3,
    )
    fig_2.add_trace(
        go.Box(
            y=total_values,
            boxmean="sd",
            name="总认可率",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
        ),
        row=2,
        col=3,
    )
    fig_2.update_layout(
        height=400 * 2,
        width=1600,
        showlegend=True,
        template="plotly_white",
    )

    figs.append(fig_2)

    # 新增：强度对比图
    if (
        "positive_intensity" in all_metrics[0]
        and "negative_intensity" in all_metrics[0]
    ):
        fig_3 = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("积极词符合程度分布", "消极词符合程度分布"),
            specs=[
                [{"type": "box"}, {"type": "box"}],
            ],
            vertical_spacing=0.12,
        )

        pos_intensity = [
            m["positive_intensity"] for m in all_metrics if "positive_intensity" in m
        ]
        neg_intensity = [
            m["negative_intensity"] for m in all_metrics if "negative_intensity" in m
        ]

        fig_3.add_trace(
            go.Box(
                y=pos_intensity,
                boxmean="sd",
                name="积极词符合程度",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                marker_color="lightblue",
            ),
            row=1,
            col=1,
        )

        fig_3.add_trace(
            go.Box(
                y=neg_intensity,
                boxmean="sd",
                name="消极词符合程度",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                marker_color="lightcoral",
            ),
            row=1,
            col=2,
        )

        figs.append(fig_3)

    # 新增：词级可视化图
    if all_results:
        all_word_stats = []
        for result in all_results:
            if "word_results" in result and "word_stats" in result["word_results"]:
                all_word_stats.append(result["word_results"]["word_stats"])

        if all_word_stats:
            word_fig = create_word_level_visualizations_group(all_word_stats)
            if word_fig:
                figs.extend(word_fig)

    return figs


def create_word_level_visualizations_multi_group(
    control_word_stats: list[list[dict]],
    experimental_word_stats: list[list[dict]],
    result_dir: Path = None,
) -> list[go.Figure]:
    """创建多组分析的词级可视化 - 横向条形图"""
    if not control_word_stats or not experimental_word_stats:
        return []

    figs = []

    # 收集所有词的数据
    def collect_group_data(word_stats_list, group_name):
        word_data = {}
        for subject_word_stats in word_stats_list:
            for word_stat in subject_word_stats:
                word = word_stat["stim_word"]
                stim_type = word_stat["stim_type"]

                if word not in word_data:
                    word_data[word] = {
                        "stim_type": stim_type,
                        "endorsement_counts": [],
                        "rejection_counts": [],
                        "mean_rts": [],
                    }

                word_data[word]["endorsement_counts"].append(
                    word_stat["endorsement_count"]
                )
                word_data[word]["rejection_counts"].append(word_stat["rejection_count"])

                if word_stat.get("mean_rt") is not None:
                    word_data[word]["mean_rts"].append(word_stat["mean_rt"])
        return word_data

    control_data = collect_group_data(control_word_stats, "control")
    experimental_data = collect_group_data(experimental_word_stats, "experimental")

    # 获取所有词的交集
    all_words = sorted(set(control_data.keys()) & set(experimental_data.keys()))

    if not all_words:
        print("⚠️ 对照组和实验组没有共同的词，无法进行词级比较分析")
        return []

    # 准备数据
    words = []
    stim_types = []
    control_endorsement_rates = []
    control_endorsement_ses = []
    experimental_endorsement_rates = []
    experimental_endorsement_ses = []
    control_mean_rts = []
    control_std_rts = []
    control_se_rts = []
    experimental_mean_rts = []
    experimental_std_rts = []
    experimental_se_rts = []

    for word in all_words:
        # 对照组数据
        ctl_endorsement_counts = control_data[word]["endorsement_counts"]
        ctl_rejection_counts = control_data[word]["rejection_counts"]
        ctl_total = np.sum(ctl_endorsement_counts) + np.sum(ctl_rejection_counts)
        ctl_endorsement_rate = (
            np.sum(ctl_endorsement_counts) / ctl_total if ctl_total > 0 else 0
        )
        ctl_n = len(ctl_endorsement_counts)
        ctl_se = (
            np.sqrt(ctl_endorsement_rate * (1 - ctl_endorsement_rate) / ctl_n)
            if ctl_n > 0 and 0 < ctl_endorsement_rate < 1
            else 0
        )

        # 实验组数据
        exp_endorsement_counts = experimental_data[word]["endorsement_counts"]
        exp_rejection_counts = experimental_data[word]["rejection_counts"]
        exp_total = np.sum(exp_endorsement_counts) + np.sum(exp_rejection_counts)
        exp_endorsement_rate = (
            np.sum(exp_endorsement_counts) / exp_total if exp_total > 0 else 0
        )
        exp_n = len(exp_endorsement_counts)
        exp_se = (
            np.sqrt(exp_endorsement_rate * (1 - exp_endorsement_rate) / exp_n)
            if exp_n > 0 and 0 < exp_endorsement_rate < 1
            else 0
        )

        # 平均反应时 - 确保有有效数据
        ctl_mean_rt = 0
        ctl_std_rt = 0
        ctl_se_rt = 0
        if control_data[word]["mean_rts"] and not all(
            np.isnan(rt) for rt in control_data[word]["mean_rts"]
        ):
            valid_rts = [
                rt for rt in control_data[word]["mean_rts"] if not np.isnan(rt)
            ]
            if valid_rts:
                ctl_mean_rt = np.mean(valid_rts)
                ctl_std_rt = np.std(valid_rts, ddof=1) if len(valid_rts) > 1 else 0
                ctl_se_rt = (
                    ctl_std_rt / np.sqrt(len(valid_rts)) if len(valid_rts) > 0 else 0
                )

        exp_mean_rt = 0
        exp_std_rt = 0
        exp_se_rt = 0
        if experimental_data[word]["mean_rts"] and not all(
            np.isnan(rt) for rt in experimental_data[word]["mean_rts"]
        ):
            valid_rts = [
                rt for rt in experimental_data[word]["mean_rts"] if not np.isnan(rt)
            ]
            if valid_rts:
                exp_mean_rt = np.mean(valid_rts)
                exp_std_rt = np.std(valid_rts, ddof=1) if len(valid_rts) > 1 else 0
                exp_se_rt = (
                    exp_std_rt / np.sqrt(len(valid_rts)) if len(valid_rts) > 0 else 0
                )

        words.append(word)
        stim_types.append(control_data[word]["stim_type"])
        control_endorsement_rates.append(ctl_endorsement_rate)
        control_endorsement_ses.append(ctl_se * 100)  # 转换为百分比
        experimental_endorsement_rates.append(exp_endorsement_rate)
        experimental_endorsement_ses.append(exp_se * 100)  # 转换为百分比
        control_mean_rts.append(ctl_mean_rt)
        control_std_rts.append(ctl_std_rt)
        control_se_rts.append(ctl_se_rt)
        experimental_mean_rts.append(exp_mean_rt)
        experimental_std_rts.append(exp_std_rt)
        experimental_se_rts.append(exp_se_rt)

    # 按词性排序
    positive_indices = [i for i, t in enumerate(stim_types) if t == "positive"]
    negative_indices = [i for i, t in enumerate(stim_types) if t == "negative"]

    sorted_indices = positive_indices + negative_indices
    words = [words[i] for i in sorted_indices]
    stim_types = [stim_types[i] for i in sorted_indices]
    control_endorsement_rates = [control_endorsement_rates[i] for i in sorted_indices]
    control_endorsement_ses = [control_endorsement_ses[i] for i in sorted_indices]
    experimental_endorsement_rates = [
        experimental_endorsement_rates[i] for i in sorted_indices
    ]
    experimental_endorsement_ses = [
        experimental_endorsement_ses[i] for i in sorted_indices
    ]
    control_mean_rts = [control_mean_rts[i] for i in sorted_indices]
    control_std_rts = [control_std_rts[i] for i in sorted_indices]
    control_se_rts = [control_se_rts[i] for i in sorted_indices]
    experimental_mean_rts = [experimental_mean_rts[i] for i in sorted_indices]
    experimental_std_rts = [experimental_std_rts[i] for i in sorted_indices]
    experimental_se_rts = [experimental_se_rts[i] for i in sorted_indices]

    # 只显示前30个词，避免图表过于拥挤
    display_limit = min(80, len(words))
    display_words = words[:display_limit]
    display_stim_types = stim_types[:display_limit]
    display_control_endorsement_rates = control_endorsement_rates[:display_limit]
    display_control_endorsement_ses = control_endorsement_ses[:display_limit]
    display_experimental_endorsement_rates = experimental_endorsement_rates[
        :display_limit
    ]
    display_experimental_endorsement_ses = experimental_endorsement_ses[:display_limit]
    display_control_mean_rts = control_mean_rts[:display_limit]
    display_control_std_rts = control_std_rts[:display_limit]
    display_control_se_rts = control_se_rts[:display_limit]
    display_experimental_mean_rts = experimental_mean_rts[:display_limit]
    display_experimental_std_rts = experimental_std_rts[:display_limit]
    display_experimental_se_rts = experimental_se_rts[:display_limit]

    # 1. 创建认同率对比图
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "组间认同率对比",
            "组间反应时对比",
            "组间认同率差异",
            "组间反应时差异",
        ),
        specs=[
            [{"type": "bar"}],
            [{"type": "bar"}],
            [{"type": "bar"}],
            [{"type": "bar"}],
        ],
        vertical_spacing=0.1,
    )

    # 添加对照组数据
    fig.add_trace(
        go.Bar(
            x=display_words,
            y=[
                rate * 100 for rate in display_control_endorsement_rates
            ],  # 转换为百分比
            name="对照组",
            marker_color=[
                "rgba(0, 128, 0, 0.7)" if t == "positive" else "rgba(255, 99, 71, 0.7)"
                for t in display_stim_types
            ],
            text=[f"{rate:.1%}" for rate in display_control_endorsement_rates],
            textposition="auto",
            error_y=dict(
                type="data",
                array=display_control_endorsement_ses,
                visible=True,
                color="rgba(0,0,0,0.5)",
            ),
            hovertemplate="词: %{x}<br>词性: %{customdata}<br>对照组认同率: %{text}<br>标准误: %{error_y.array:.1f}%<br><extra></extra>",
            customdata=display_stim_types,
        ),
        row=1,
        col=1,
    )

    # 添加实验组数据
    fig.add_trace(
        go.Bar(
            x=display_words,
            y=[
                rate * 100 for rate in display_experimental_endorsement_rates
            ],  # 转换为百分比
            name="实验组",
            marker_color=[
                "rgba(144, 238, 144, 0.7)"
                if t == "positive"
                else "rgba(255, 182, 193, 0.7)"
                for t in display_stim_types
            ],
            text=[f"{rate:.1%}" for rate in display_experimental_endorsement_rates],
            textposition="auto",
            error_y=dict(
                type="data",
                array=display_experimental_endorsement_ses,
                visible=True,
                color="rgba(0,0,0,0.5)",
            ),
            hovertemplate="词: %{x}<br>词性: %{customdata}<br>实验组认同率: %{text}<br>标准误: %{error_y.array:.1f}%<br><extra></extra>",
            customdata=display_stim_types,
        ),
        row=1,
        col=1,
    )

    # 更新x轴标签角度
    fig.update_xaxes(tickangle=45, row=1, col=1)

    # 2. 创建反应时对比图 - 确保有有效数据
    if any(rt > 0 for rt in display_control_mean_rts) or any(
        rt > 0 for rt in display_experimental_mean_rts
    ):
        # 添加对照组数据
        fig.add_trace(
            go.Bar(
                x=display_words,
                y=display_control_mean_rts,
                name="对照组",
                marker_color=[
                    "rgba(0, 128, 0, 0.7)"
                    if t == "positive"
                    else "rgba(255, 99, 71, 0.7)"
                    for t in display_stim_types
                ],
                text=[f"{rt:.0f}ms" for rt in display_control_mean_rts],
                textposition="auto",
                error_y=dict(
                    type="data",
                    array=display_control_std_rts,
                    visible=True,
                    color="rgba(0,0,0,0.5)",
                ),
                hovertemplate="词: %{x}<br>词性: %{customdata}<br>对照组反应时: %{text}<br>标准差: %{error_y.array:.0f}ms<br><extra></extra>",
                customdata=display_stim_types,
            ),
            row=2,
            col=1,
        )

        # 添加实验组数据
        fig.add_trace(
            go.Bar(
                x=display_words,
                y=display_experimental_mean_rts,
                name="实验组",
                marker_color=[
                    "rgba(144, 238, 144, 0.7)"
                    if t == "positive"
                    else "rgba(255, 182, 193, 0.7)"
                    for t in display_stim_types
                ],
                text=[f"{rt:.0f}ms" for rt in display_experimental_mean_rts],
                textposition="auto",
                error_y=dict(
                    type="data",
                    array=display_experimental_std_rts,
                    visible=True,
                    color="rgba(0,0,0,0.5)",
                ),
                hovertemplate="词: %{x}<br>词性: %{customdata}<br>实验组反应时: %{text}<br>标准差: %{error_y.array:.0f}ms<br><extra></extra>",
                customdata=display_stim_types,
            ),
            row=2,
            col=1,
        )

        # 更新x轴标签角度
        fig.update_xaxes(tickangle=45, row=2, col=1)

    # 3. 创建认同率差异图

    endorsement_rate_diffs = [
        (exp - ctl) * 100
        for exp, ctl in zip(
            display_experimental_endorsement_rates, display_control_endorsement_rates
        )
    ]

    # 计算差异的标准误（假设独立）
    diff_ses = []
    for i in range(len(display_words)):
        se_ctl = display_control_endorsement_ses[i] / 100  # 转换回比例
        se_exp = display_experimental_endorsement_ses[i] / 100  # 转换回比例
        # 差异的标准误 = sqrt(se_ctl^2 + se_exp^2)
        diff_se = np.sqrt(se_ctl**2 + se_exp**2) * 100  # 转换回百分比
        diff_ses.append(diff_se)

    fig.add_trace(
        go.Bar(
            x=display_words,
            y=endorsement_rate_diffs,
            name="认同率差异",
            marker_color=[
                "rgba(0, 0, 255, 0.7)" if diff > 0 else "rgba(255, 0, 0, 0.7)"
                for diff in endorsement_rate_diffs
            ],
            error_y=dict(
                type="data", array=diff_ses, visible=True, color="rgba(0,0,0,0.5)"
            ),
            hovertemplate="词: %{x}<br>词性: %{customdata}<br>认同率差异(实验-对照): %{text}%<br>标准误: %{error_y.array:.1f}%<br><extra></extra>",
            customdata=display_stim_types,
            text=[diff for diff in endorsement_rate_diffs],
            textposition="none",
        ),
        row=3,
        col=1,
    )

    # 添加零线
    fig.add_hline(
        y=0,
        line_width=2,
        line_dash="dash",
        line_color="gray",
        annotation_text="零差异线",
        annotation_position="right",
        row=3,
        col=1,
    )

    # 更新x轴标签角度
    fig.update_xaxes(tickangle=45, row=3, col=1)
    # 4. 新增：反应时差异分析图
    # 计算每词的反应时差异（实验组 - 对照组）
    rt_diffs = [
        exp - ctl
        for exp, ctl in zip(display_experimental_mean_rts, display_control_mean_rts)
    ]

    # 计算差异的标准误（简化计算）
    rt_diff_ses = []
    for i in range(len(display_words)):
        se_ctl = display_control_se_rts[i]
        se_exp = display_experimental_se_rts[i]
        diff_se = np.sqrt(se_ctl**2 + se_exp**2)
        rt_diff_ses.append(diff_se)

    fig.add_trace(
        go.Bar(
            x=display_words,
            y=rt_diffs,
            name="反应时差异",
            marker_color=[
                "rgba(0, 0, 255, 0.7)" if diff > 0 else "rgba(255, 0, 0, 0.7)"
                for diff in rt_diffs
            ],
            error_y=dict(
                type="data", array=rt_diff_ses, visible=True, color="rgba(0,0,0,0.5)"
            ),
            hovertemplate="词: %{x}<br>词性: %{customdata}<br>反应时差异(实验-对照): %{text:.0f}ms<br>标准误: %{error_y.array:.0f}ms<br><extra></extra>",
            customdata=display_stim_types,
            text=[diff for diff in rt_diffs],
            textposition="none",
        ),
        row=4,
        col=1,
    )

    # 添加零线
    fig.add_hline(
        y=0,
        line_width=2,
        line_dash="dash",
        line_color="gray",
        annotation_text="零差异线",
        annotation_position="right",
        row=4,
        col=1,
    )

    # 更新x轴标签角度
    fig.update_xaxes(tickangle=45, row=4, col=1)
    if result_dir:
        # 保存完整的词级数据到CSV
        word_stats_df = pd.DataFrame(
            {
                "word": words,
                "stim_type": stim_types,
                "control_endorsement_rate": control_endorsement_rates,
                "control_endorsement_se(%)": control_endorsement_ses,
                "experimental_endorsement_rate": experimental_endorsement_rates,
                "experimental_endorsement_se(%)": experimental_endorsement_ses,
                "endorsement_rate_diff(%)": [
                    exp * 100 - ctl * 100
                    for exp, ctl in zip(
                        experimental_endorsement_rates, control_endorsement_rates
                    )
                ],
                "control_mean_rt": control_mean_rts,
                "control_std_rt": control_std_rts,
                "experimental_mean_rt": experimental_mean_rts,
                "experimental_std_rt": experimental_std_rts,
                "rt_diff": [
                    exp - ctl
                    for exp, ctl in zip(experimental_mean_rts, control_mean_rts)
                ],
            }
        )
        word_stats_df.to_csv(
            result_dir / "sret_multi_group_word_level_stats.csv", index=False
        )
    # 积极消极分界线
    fig.add_vline(
        x=len(positive_indices) - 0.5,
        line_width=2,
        line_dash="dash",
        line_color="gray",
    )

    fig.update_layout(
        title=dict(
            text=f"SRET多组分析词级报告 - (显示前{display_limit}个词)",
            font=dict(size=16, family="Arial Black"),
            x=0.5,
        ),
        height=600 * 4,
        width=2000,
        barmode="group",
        template="plotly_white",
        hovermode="x unified",
    )

    figs.append(fig)
    return figs


def create_multi_group_visualizations(
    control_metrics: list[dict[str, float]],
    experimental_metrics: list[dict[str, float]],
    control_results: list[dict] | None = None,
    experimental_results: list[dict] | None = None,
) -> list[go.Figure]:
    """多组分析可视化 - 使用all_metrics中的所有指标"""
    figs = []

    # 原有的可视化图
    # 收集所有可用的指标
    all_available_metrics = []
    for metric in all_metrics:
        if metric in control_metrics[0] and metric in experimental_metrics[0]:
            all_available_metrics.append(metric)

    # 创建多个图表
    fig_bias = make_subplots(
        rows=1, cols=1, subplot_titles=("积极偏向对比",), specs=[[{"type": "bar"}]]
    )

    fig_rt = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "消极-积极RT差",
            "认同-不认同RT差",
            "积极词RT",
            "消极词RT",
            "认同RT",
            "不认同RT",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
        ],
    )

    fig_endorsement = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("积极认可率", "消极认可率", "总认可率"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
    )

    fig_intensity = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("积极词符合程度", "消极词符合程度", "消极-积极符合程度差"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
    )

    # 辅助函数：添加柱状图对比
    def add_bar_comparison(
        fig, metric_name, display_name, row, col, control_metrics, experimental_metrics
    ):
        control_values = [m[metric_name] for m in control_metrics if metric_name in m]
        experimental_values = [
            m[metric_name] for m in experimental_metrics if metric_name in m
        ]

        if control_values and experimental_values:
            control_mean = np.mean(control_values)
            control_std = (
                np.std(control_values, ddof=1) if len(control_values) > 1 else 0
            )
            experimental_mean = np.mean(experimental_values)
            experimental_std = (
                np.std(experimental_values, ddof=1)
                if len(experimental_values) > 1
                else 0
            )

            fig.add_trace(
                go.Bar(
                    x=["对照组", "实验组"],
                    y=[control_mean, experimental_mean],
                    name=display_name,
                    marker_color=["green", "red"],
                    error_y=dict(
                        type="data", array=[control_std, experimental_std], visible=True
                    ),
                    text=[f"{control_mean:.3f}", f"{experimental_mean:.3f}"],
                    textposition="auto",
                ),
                row=row,
                col=col,
            )

    # 添加积极偏向对比
    add_bar_comparison(
        fig_bias,
        "positive_bias",
        "积极偏向",
        1,
        1,
        control_metrics,
        experimental_metrics,
    )

    # 添加反应时指标对比
    rt_metrics_list = [
        ("rt_negative_minus_positive", "消极-积极RT差", 1, 1),
        ("rt_endorsed_minus_not", "认同-不认同RT差", 1, 2),
        ("positive_rt", "积极词RT", 1, 3),
        ("negative_rt", "消极词RT", 2, 1),
        ("yes_rt", "认同RT", 2, 2),
        ("no_rt", "不认同RT", 2, 3),
    ]

    for metric, display_name, row, col in rt_metrics_list:
        add_bar_comparison(
            fig_rt,
            metric,
            display_name,
            row,
            col,
            control_metrics,
            experimental_metrics,
        )

    # 添加认可率指标对比
    endorsement_metrics_list = [
        ("positive_endorsement_rate", "积极认可率", 1, 1),
        ("negative_endorsement_rate", "消极认可率", 1, 2),
        ("endorsement_rate", "总认可率", 1, 3),
    ]

    for metric, display_name, row, col in endorsement_metrics_list:
        add_bar_comparison(
            fig_endorsement,
            metric,
            display_name,
            row,
            col,
            control_metrics,
            experimental_metrics,
        )

    # 添加强度指标对比
    intensity_metrics_list = [
        ("positive_intensity", "积极词符合程度", 1, 1),
        ("negative_intensity", "消极词符合程度", 1, 2),
        ("intensity_negative_minus_positive", "消极-积极符合程度差", 1, 3),
    ]

    for metric, display_name, row, col in intensity_metrics_list:
        add_bar_comparison(
            fig_intensity,
            metric,
            display_name,
            row,
            col,
            control_metrics,
            experimental_metrics,
        )

    # 更新所有图的布局
    fig_bias.update_layout(
        height=400,
        title=dict(text="积极偏向对比", font=dict(size=16)),
        showlegend=False,
        template="plotly_white",
    )

    fig_rt.update_layout(
        height=400 * 2,
        width=1600,
        title=dict(text="反应时指标对比", font=dict(size=16)),
        showlegend=False,
        template="plotly_white",
    )

    fig_endorsement.update_layout(
        height=400,
        width=1600,
        title=dict(text="认可率指标对比", font=dict(size=16)),
        showlegend=False,
        template="plotly_white",
    )

    fig_intensity.update_layout(
        height=400,
        width=1600,
        title=dict(text="符合程度指标对比", font=dict(size=16)),
        showlegend=False,
        template="plotly_white",
    )

    # 将所有图添加到返回列表
    figs.extend([fig_bias, fig_rt, fig_endorsement, fig_intensity])

    # 新增：词级可视化图
    if control_results and experimental_results:
        control_word_stats = []
        experimental_word_stats = []

        for result in control_results:
            if "word_results" in result and "word_stats" in result["word_results"]:
                control_word_stats.append(result["word_results"]["word_stats"])

        for result in experimental_results:
            if "word_results" in result and "word_stats" in result["word_results"]:
                experimental_word_stats.append(result["word_results"]["word_stats"])

        if control_word_stats and experimental_word_stats:
            word_fig = create_word_level_visualizations_multi_group(
                control_word_stats, experimental_word_stats
            )
            if word_fig:
                figs.extend(word_fig)

    return figs


def run_single_sret_analysis(file_path: Path, result_dir: Path = None):
    """单个被试的SRET分析"""

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    if result_dir is None:
        result_dir = file_path.parent / "sret_results"

    result_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(file_path)

    result = analyze_sret_data_single(
        df=df, target_blocks=["Encoding"], result_dir=result_dir
    )

    print(f"\n✅ 分析完成！结果保存在: {result_dir}")
    return result


def run_group_sret_analysis(
    data_files: list[Path],
    result_dir: Path = None,
    group_name: str = None,
    reference_group: Literal["control", "mdd"] = None,
):
    """组SRET分析"""
    if result_dir is None:
        result_dir = Path("sret_group_results")

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

            result = analyze_sret_data_single(
                df=df,
                target_blocks=["Encoding"],
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
        print("1. 对照组")
        print("2. mdd组")
        choice = input("选择 (1/2): ").strip()

        reference_group = "control" if choice == "1" else "mdd"

    # 单样本t检验 - 仍使用key_metrics
    statistical_results = {}

    # ref value 计算
    ctl_positive_endorsed = REFERENCE_VALUES["control"]["endorsement_count"]["positive"]
    ctl_negative_endorsed = REFERENCE_VALUES["control"]["endorsement_count"]["negative"]

    mdd_positive_endorsed = REFERENCE_VALUES["mdd"]["endorsement_count"]["positive"]
    mdd_negative_endorsed = REFERENCE_VALUES["mdd"]["endorsement_count"]["negative"]

    for metric in key_metrics:
        # 获取当前组的指标值
        group_values = [m[metric] for m in group_metrics if metric in m]

        if len(group_values) < 2:
            statistical_results[metric] = {"error": "样本量不足"}
            continue

        if metric == "positive_bias":
            if reference_group == "control":
                positive_rate = ctl_positive_endorsed / 40
                negative_rate = ctl_negative_endorsed / 40
                ref_value = positive_rate - negative_rate
            else:
                positive_rate = mdd_positive_endorsed / 40
                negative_rate = mdd_negative_endorsed / 40
                ref_value = positive_rate - negative_rate
        elif metric == "rt_negative_minus_positive":
            if reference_group == "control":
                ref_value = (
                    REFERENCE_VALUES["control"]["reaction_time"]["negative"]
                    - REFERENCE_VALUES["control"]["reaction_time"]["positive"]
                )
            else:
                ref_value = (
                    REFERENCE_VALUES["mdd"]["reaction_time"]["negative"]
                    - REFERENCE_VALUES["mdd"]["reaction_time"]["positive"]
                )
        elif metric == "rt_endorsed_minus_not":
            ref_value = np.mean(group_values) * 0.9
        elif metric == "endorsement_rate":
            # 估算总认同率
            if reference_group == "control":
                positive_endorsed = ctl_positive_endorsed
                negative_endorsed = ctl_negative_endorsed
                ref_value = (positive_endorsed + negative_endorsed) / 80
            else:
                positive_endorsed = mdd_positive_endorsed
                negative_endorsed = mdd_negative_endorsed
                ref_value = (positive_endorsed + negative_endorsed) / 80
        elif metric == "positive_endorsement_rate":
            if reference_group == "control":
                positive_endorsed = ctl_positive_endorsed
                ref_value = positive_endorsed / 40
            else:
                positive_endorsed = mdd_positive_endorsed
                ref_value = positive_endorsed / 40
        elif metric == "negative_endorsement_rate":
            if reference_group == "control":
                negative_endorsed = ctl_negative_endorsed
                ref_value = negative_endorsed / 40
            else:
                negative_endorsed = mdd_negative_endorsed
                ref_value = negative_endorsed / 40

        # 单样本t检验
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

    fig_spec = create_single_group_visualizations(group_metrics, all_results)

    fig_common = create_common_single_group_figures(
        group_metrics, statistical_results, key_metrics, metric_names
    )

    figs = fig_spec + fig_common

    save_html_report(
        result_dir,
        f"sret-{group_name}_group-analysis_report",
        figs,
        title=f"SRET{group_name}组分析报告",
    )

    return {
        "all_results": all_results,
        "group_metrics": group_metrics,
        "statistical_results": statistical_results,
        "group_mean": group_mean_metrics,
        "group_std": group_std_metrics,
    }


def run_groups_sret_analysis(
    control_files: list[Path],
    experimental_files: list[Path],
    result_dir: Path = Path("group_comparison_results"),
    groups: list[str] = None,
) -> dict[str, Any]:
    """比较对照组和实验组 - 使用all_metrics进行组间比较"""

    control_name = groups[0] if groups else "control"

    control_group_results = run_group_sret_analysis(
        control_files, result_dir, control_name, reference_group="control"
    )
    control_results = control_group_results["all_results"]
    control_metrics = control_group_results["group_metrics"]

    experimental_name = groups[1] if groups and len(groups) > 1 else "experimental"

    experimental_group_results = run_group_sret_analysis(
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
        control_metrics + experimental_metrics, all_metrics
    )

    print("\n执行组间比较分析...")
    comparison_results = perform_group_comparisons(
        control_metrics, experimental_metrics, all_metrics
    )

    print("\n保存组分析结果...")

    all_control_metrics_df = pd.DataFrame([r["key_metrics"] for r in control_results])
    all_control_metrics_df.insert(0, "group", control_name)

    all_experimental_metrics_df = pd.DataFrame(
        [r["key_metrics"] for r in experimental_results]
    )
    all_experimental_metrics_df.insert(0, "group", experimental_name)

    all_metrics_df = pd.concat(
        [all_control_metrics_df, all_experimental_metrics_df], ignore_index=True
    )
    all_metrics_df.to_csv(result_dir / "all_subjects_metrics.csv", index=False)

    control_stats = all_control_metrics_df.drop(columns=["group"]).describe()
    experimental_stats = all_experimental_metrics_df.drop(columns=["group"]).describe()

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

    if normality_results:
        normality_df = pd.DataFrame(normality_results).T
        normality_df.to_csv(result_dir / "normality_tests.csv")

    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results).T
        comparison_df.to_csv(result_dir / "group_comparisons.csv")

    fig_spec = create_multi_group_visualizations(
        control_metrics, experimental_metrics, control_results, experimental_results
    )

    fig_common = create_common_comparison_figures(
        comparison_results, all_metrics, all_metric_names
    )

    figs = fig_spec + fig_common
    save_html_report(
        save_dir=result_dir,
        save_name=f"sret-{control_name}_{experimental_name}_group-comparison_report",
        figures=figs,
        title=f"SRET{control_name}-{experimental_name}组间比较分析",
    )

    return {
        "control_results": control_results,
        "experimental_results": experimental_results,
        "control_metrics": control_metrics,
        "experimental_metrics": experimental_metrics,
        "normality_results": normality_results,
        "comparison_results": comparison_results,
    }


def run_sret_analysis(cfg: DictConfig = None, data_utils: DataUtils = None):
    """SRET（自我参照编码任务）分析入口函数"""

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

            result_dir = result_root / "sret_results"
            result_dir.mkdir(parents=True, exist_ok=True)

            result = run_single_sret_analysis(file_path, result_dir)
            return result

        elif choice == "2":
            dir_input = input("请输入包含多个数据文件的目录路径: ").strip("'").strip()
            data_dir = Path(dir_input.strip("'").strip('"')).resolve()

            if not data_dir.exists():
                print(f"❌ 目录不存在: {data_dir}")
                return

            data_files = find_sret_files(data_dir)
            print(f"找到 {len(data_files)} 个数据文件")

            result_dir = result_root / "sret_group_results"
            result_dir.mkdir(parents=True, exist_ok=True)

            result = run_group_sret_analysis(data_files, result_dir)
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

            control_files = find_sret_files(control_dir)
            experimental_files = find_sret_files(experimental_dir)

            print(
                f"找到 {len(control_files)} 个对照组文件和 {len(experimental_files)} 个实验组文件"
            )

            result_dir = result_root / "sret_group_comparison_results"
            result_dir.mkdir(parents=True, exist_ok=True)

            result = run_groups_sret_analysis(
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
                files = find_sret_files(data_root / groups[0])
                result_dir = result_root / f"sret_{groups[0]}_results"
                result_dir.mkdir(parents=True, exist_ok=True)

                reference_group = "mdd" if "mdd" in groups[0].lower() else "control"
                run_group_sret_analysis(
                    files, result_dir, reference_group=reference_group
                )
            else:
                # 多个组分析
                control_files = find_sret_files(data_root / groups[0])
                experimental_files = find_sret_files(data_root / groups[1])
                result_dir = (
                    result_root / f"sret_{groups[0]}_{groups[1]}_comparison_results"
                )
                result_dir.mkdir(parents=True, exist_ok=True)
                run_groups_sret_analysis(
                    control_files, experimental_files, result_dir, groups
                )
        else:
            # 单个被试分析
            file_path = (
                Path(cfg.output_dir)
                / data_utils.date
                / f"{data_utils.session_id}-sret.csv"
            )

            if not file_path.exists():
                print(f"❌ 文件不存在: {file_path}")
                return

            result_dir = result_root / str(data_utils.session_id) / "sret_analysis"
            result_dir.mkdir(parents=True, exist_ok=True)

            result = run_single_sret_analysis(file_path, result_dir)
            return result


if __name__ == "__main__":
    run_sret_analysis()
