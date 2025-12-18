import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from omegaconf import DictConfig
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

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

# 核心关键指标 - 单人/单组分析使用
key_metrics = [
    "overall_accuracy",
    "positive_accuracy",
    "negative_accuracy",
    "neutral_accuracy",
    "positive_neg_rt_diff",
]

metric_names = [
    "总体正确率",
    "积极正确率",
    "消极正确率",
    "中性正确率",
    "积极-消极反应时差",
]

# 全局所有指标 - 多组分析使用
all_metrics = [
    # 正确率指标
    "overall_accuracy",
    "positive_accuracy",
    "negative_accuracy",
    "neutral_accuracy",
    # 反应时指标
    "positive_rt",
    "negative_rt",
    "neutral_rt",
    "positive_neg_rt_diff",
    # 强度指标
    "positive_intensity",
    "negative_intensity",
    "mean_intensity_diff",
    "intensity_correlation_r",
    # 中性阈值指标
    "neutral_choice_count",
    "neutral_mean_label_intensity",
]

all_metric_names = [
    # 正确率指标
    "总体正确率",
    "积极正确率",
    "消极正确率",
    "中性正确率",
    # 反应时指标
    "积极反应时",
    "消极反应时",
    "中性反应时",
    "积极-消极反应时差",
    # 强度指标
    "积极强度评分",
    "消极强度评分",
    "平均强度差异",  # 平均强度差异 = 选择的强度评分 - 实际强度评分
    "强度相关性r值",  # 强度相关性r值 = 所选强度评分与实际强度评分的相关系数
    # 中性阈值指标
    "被判断为中性的次数",
    "中性阈值平均标签强度",
]


def find_emotion_face_files(data_dir: Path) -> list[Path]:
    """查找指定目录下的面部情绪识别实验结果文件"""
    EXP_TYPE = "face_recognition"
    return find_exp_files(data_dir, EXP_TYPE)


def load_and_preprocess_data(df: pl.DataFrame) -> pl.DataFrame:
    """加载并预处理面部情绪识别数据"""
    try:
        trials_df = extract_trials_by_block(
            df,
            target_block_indices=[0, 1],
            block_col="block_index",
            trial_col="trial_index",
        )

        if trials_df.height == 0:
            print("❌ 错误: 未找到有效的试次数据")
            return None

        def map_stim_type(col_value):
            if col_value is None or col_value == "" or col_value == "null":
                return "unknown"
            stim_lower = str(col_value).lower()
            if "positive" in stim_lower or "pos" in stim_lower:
                return "positive"
            elif "negative" in stim_lower or "neg" in stim_lower:
                return "negative"
            elif "neutral" in stim_lower or "neu" in stim_lower:
                return "neutral"
            else:
                return "unknown"

        trials_df = trials_df.with_columns(
            [
                pl.col("stim")
                .map_elements(map_stim_type, return_dtype=pl.Utf8, skip_nulls=False)
                .alias("stim_type"),
                pl.col("choice")
                .map_elements(map_stim_type, return_dtype=pl.Utf8, skip_nulls=False)
                .alias("choice_type"),
            ]
        )

        # 添加强度等级划分
        def get_intensity_level(label_intensity):
            if label_intensity is None:
                return None
            if 1 <= label_intensity <= 3:
                return "low"
            elif 4 <= label_intensity <= 6:
                return "mid"
            elif 7 <= label_intensity <= 9:
                return "high"
            else:
                return None

        trials_df = trials_df.with_columns(
            pl.col("label_intensity")
            .map_elements(get_intensity_level, return_dtype=pl.Utf8, skip_nulls=False)
            .alias("intensity_level")
        )

        # 去除过大过小而不是强制转换
        trials_df = trials_df.with_columns(
            pl.when(pl.col("rt") < 0.1)
            .then(0.1)
            .when(pl.col("rt") > 5.0)
            .then(5.0)
            .when(pl.col("rt").is_null())
            .then(pl.col("rt").max() + 0.6)
            .otherwise(pl.col("rt"))
            .alias("rt_clean")
        )

        trials_df = trials_df.with_columns(
            (pl.col("stim_type") == pl.col("choice_type")).alias("correct")
        )

        return trials_df

    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
        return None


def calculate_intensity_distribution_metrics(trials_df: pl.DataFrame) -> dict[str, Any]:
    """计算强度评分分布指标"""
    metrics = {}

    # 过滤非中性刺激（中性刺激通常没有强度评分）
    intensity_trials = trials_df.filter(pl.col("stim_type") != "neutral")

    if intensity_trials.height == 0:
        metrics["has_intensity_data"] = False
        return metrics

    metrics["has_intensity_data"] = True

    # 基本分布统计
    intensity_values = intensity_trials["intensity"].drop_nulls().to_list()
    label_intensity_values = intensity_trials["label_intensity"].drop_nulls().to_list()

    if intensity_values:
        metrics["intensity_distribution"] = {
            "mean": float(np.mean(intensity_values)),
            "std": float(np.std(intensity_values, ddof=1)),
            "median": float(np.median(intensity_values)),
            "min": float(np.min(intensity_values)),
            "max": float(np.max(intensity_values)),
            "n": len(intensity_values),
        }

    if label_intensity_values:
        metrics["label_intensity_distribution"] = {
            "mean": float(np.mean(label_intensity_values)),
            "std": float(np.std(label_intensity_values, ddof=1)),
            "median": float(np.median(label_intensity_values)),
            "min": float(np.min(label_intensity_values)),
            "max": float(np.max(label_intensity_values)),
            "n": len(label_intensity_values),
        }

    # 按情绪类型的强度分布
    intensity_by_emotion = (
        intensity_trials.group_by("stim_type")
        .agg(
            [
                pl.col("intensity").mean().alias("mean_intensity"),
                pl.col("intensity").std().alias("std_intensity"),
                pl.col("intensity").median().alias("median_intensity"),
                pl.col("intensity").count().alias("n_trials"),
                pl.col("label_intensity").mean().alias("mean_label_intensity"),
                pl.col("label_intensity").std().alias("std_label_intensity"),
                pl.col("label_intensity").median().alias("median_label_intensity"),
            ]
        )
        .sort("stim_type")
    )
    metrics["intensity_by_emotion"] = intensity_by_emotion

    # 强度差异分析（被试评分 vs 标签强度）
    if len(intensity_values) > 0 and len(label_intensity_values) > 0:
        # 计算差异
        intensity_diffs = intensity_trials.with_columns(
            (pl.col("intensity") - pl.col("label_intensity")).alias("intensity_diff"),
            (pl.col("intensity") - pl.col("label_intensity"))
            .abs()
            .alias("abs_intensity_diff"),
        )

        diff_stats = intensity_diffs.select(
            [
                pl.col("intensity_diff").mean().alias("mean_diff"),
                pl.col("intensity_diff").std().alias("std_diff"),
                pl.col("abs_intensity_diff").mean().alias("mean_abs_diff"),
                pl.col("abs_intensity_diff").std().alias("std_abs_diff"),
                pl.col("intensity_diff").count().alias("n"),
            ]
        )

        metrics["intensity_difference"] = {
            "mean_diff": float(diff_stats["mean_diff"][0]),
            "std_diff": float(diff_stats["std_diff"][0]),
            "mean_abs_diff": float(diff_stats["mean_abs_diff"][0]),
            "std_abs_diff": float(diff_stats["std_abs_diff"][0]),
            "n": int(diff_stats["n"][0]),
        }

        # 按情绪类型的差异
        diff_by_emotion = (
            intensity_diffs.group_by("stim_type")
            .agg(
                [
                    pl.col("intensity_diff").mean().alias("mean_diff"),
                    pl.col("intensity_diff").std().alias("std_diff"),
                    pl.col("abs_intensity_diff").mean().alias("mean_abs_diff"),
                    pl.col("intensity_diff").count().alias("n"),
                ]
            )
            .sort("stim_type")
        )
        metrics["intensity_diff_by_emotion"] = diff_by_emotion

        # 强度相关性
        corr_data = intensity_trials.filter(
            pl.col("label_intensity").is_not_null() & pl.col("intensity").is_not_null()
        )

        if corr_data.height >= 3:
            corr_df = corr_data.select(["label_intensity", "intensity"]).to_pandas()
            corr, p_val = stats.pearsonr(
                corr_df["label_intensity"], corr_df["intensity"]
            )
            metrics["intensity_correlation"] = {
                "r": float(corr),
                "p": float(p_val),
                "n": len(corr_df),
            }

    return metrics


def calculate_basic_metrics(trials_df: pl.DataFrame) -> dict[str, Any]:
    """计算基本行为指标"""
    metrics = {}

    metrics["overall_accuracy"] = trials_df["correct"].mean()
    metrics["median_rt"] = trials_df["rt_clean"].median()
    metrics["total_trials"] = trials_df.height

    block_correct = (
        trials_df.group_by("block_index")
        .agg(pl.col("correct").mean().alias("correct_rate"))
        .sort("block_index")
    )
    metrics["block_accuracy"] = block_correct

    emotion_correct = (
        trials_df.group_by("stim_type")
        .agg(
            [
                pl.col("correct").mean().alias("correct_rate"),
                pl.col("correct").count().alias("trial_count"),
            ]
        )
        .sort("stim_type")
    )
    metrics["emotion_accuracy"] = emotion_correct

    rt_summary = (
        trials_df.group_by("stim_type")
        .agg(
            [
                pl.col("rt_clean").mean().alias("mean_rt"),
                pl.col("rt_clean").std().alias("std_rt"),
                pl.col("rt_clean").median().alias("median_rt"),
                pl.col("rt_clean").count().alias("trial_count"),
            ]
        )
        .sort("stim_type")
    )
    metrics["reaction_time"] = rt_summary

    # 计算强度评分分布指标
    intensity_metrics = calculate_intensity_distribution_metrics(trials_df)
    metrics.update(intensity_metrics)

    return metrics


def calculate_intensity_metrics(trials_df: pl.DataFrame) -> dict[str, Any]:
    """计算强度指标和中性阈值"""
    metrics = {}

    # 中性阈值分析
    neutral_choices_by_stim = trials_df.filter(pl.col("choice_type") == "neutral")

    if neutral_choices_by_stim.height > 0:
        neutral_threshold_by_stim = (
            neutral_choices_by_stim.group_by("stim_type")
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col("label_intensity").mean().alias("mean_label_intensity"),
                    pl.col("label_intensity").std().alias("std_label_intensity"),
                    pl.col("label_intensity").min().alias("min_label_intensity"),
                    pl.col("label_intensity").max().alias("max_label_intensity"),
                    pl.col("label_intensity").median().alias("median_label_intensity"),
                ]
            )
            .sort("stim_type")
        )
        metrics["neutral_threshold_by_stimulus"] = neutral_threshold_by_stim

    return metrics


def perform_statistical_tests(trials_df: pl.DataFrame) -> dict[str, Any]:
    """执行统计检验"""
    results = {}

    # 转换为pandas以兼容统计库
    df_pd = trials_df.to_pandas()

    # 不同情绪类型正确率的卡方检验
    contingency_table = pd.crosstab(df_pd["stim_type"], df_pd["correct"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    results["chi2_test_accuracy"] = {"chi2": chi2, "p": p, "df": dof}

    # 反应时的方差分析
    anova_data = df_pd[["stim_type", "rt_clean"]].dropna()
    model = ols("rt_clean ~ C(stim_type)", data=anova_data).fit()
    anova_table = anova_lm(model)
    results["anova_rt"] = {
        "F": anova_table["F"][0],
        "p": anova_table["PR(>F)"][0],
        "df_num": anova_table["df"][0],
        "df_den": anova_table["df"][1],
    }

    # 速度-准确性权衡分析
    df_pd["rt_quartile"] = pd.qcut(
        df_pd["rt_clean"], 4, labels=["最快", "较快", "较慢", "最慢"]
    )
    speed_accuracy = df_pd.groupby("rt_quartile")["correct"].mean().reset_index()
    results["speed_accuracy_tradeoff"] = speed_accuracy

    return results


def calculate_comprehensive_key_metrics(
    trials_df: pl.DataFrame, metrics: dict[str, Any]
) -> dict[str, float]:
    """计算全面的关键指标"""
    key_metrics_dict = {}

    # 基本正确率指标
    key_metrics_dict["overall_accuracy"] = float(metrics["overall_accuracy"])

    # 情绪类型正确率
    emotion_acc = metrics["emotion_accuracy"]
    for stim_type in ["positive", "negative", "neutral"]:
        stim_data = emotion_acc.filter(pl.col("stim_type") == stim_type)
        if stim_data.height > 0:
            key_metrics_dict[f"{stim_type}_accuracy"] = float(
                stim_data["correct_rate"][0]
            )

    # 反应时数据
    rt_data = metrics["reaction_time"]
    key_metrics_dict["median_rt"] = float(metrics["median_rt"])

    if rt_data.height >= 2:
        positive_rt = rt_data.filter(pl.col("stim_type") == "positive")
        negative_rt = rt_data.filter(pl.col("stim_type") == "negative")
        neutral_rt = rt_data.filter(pl.col("stim_type") == "neutral")

        if positive_rt.height > 0:
            key_metrics_dict["positive_rt"] = float(positive_rt["mean_rt"][0])

        if negative_rt.height > 0:
            key_metrics_dict["negative_rt"] = float(negative_rt["mean_rt"][0])

        if neutral_rt.height > 0:
            key_metrics_dict["neutral_rt"] = float(neutral_rt["mean_rt"][0])

        if positive_rt.height > 0 and negative_rt.height > 0:
            key_metrics_dict["positive_neg_rt_diff"] = float(
                positive_rt["mean_rt"][0] - negative_rt["mean_rt"][0]
            )

    # 强度指标
    if "intensity_by_emotion" in metrics:
        intensity_by_emotion = metrics["intensity_by_emotion"]
        for stim_type in ["positive", "negative"]:
            stim_data = intensity_by_emotion.filter(pl.col("stim_type") == stim_type)
            if stim_data.height > 0:
                key_metrics_dict[f"{stim_type}_intensity"] = float(
                    stim_data["mean_intensity"][0]
                )

    # 强度差异指标
    if "intensity_difference" in metrics:
        intensity_diff = metrics["intensity_difference"]
        key_metrics_dict["mean_intensity_diff"] = intensity_diff["mean_diff"]

    # 强度相关性
    if "intensity_correlation" in metrics:
        corr = metrics["intensity_correlation"]
        key_metrics_dict["intensity_correlation_r"] = corr["r"]

    # 中性阈值指标
    if "neutral_threshold_by_stimulus" in metrics:
        neutral_threshold = metrics["neutral_threshold_by_stimulus"]
        # 计算所有刺激类型被判断为中性的总次数
        total_count = neutral_threshold["count"].sum()
        key_metrics_dict["neutral_choice_count"] = float(total_count)

        # 计算平均标签强度
        if total_count > 0:
            weighted_mean = (
                neutral_threshold["count"] * neutral_threshold["mean_label_intensity"]
            ).sum() / total_count
            key_metrics_dict["neutral_mean_label_intensity"] = float(weighted_mean)

    # 处理NaN值，用0替代
    for key in key_metrics_dict:
        if isinstance(key_metrics_dict[key], float) and np.isnan(key_metrics_dict[key]):
            key_metrics_dict[key] = 0.0

    return key_metrics_dict


def calculate_key_metrics(
    trials_df: pl.DataFrame, metrics: dict[str, Any]
) -> dict[str, float]:
    """计算关键指标（精简版，用于单人/单组分析）"""
    # 计算全面的指标
    comprehensive_metrics = calculate_comprehensive_key_metrics(trials_df, metrics)

    # 只保留核心指标
    key_metrics_dict = {}
    for metric in key_metrics:
        if metric in comprehensive_metrics:
            key_metrics_dict[metric] = comprehensive_metrics[metric]

    return key_metrics_dict


def create_visualizations(
    trials_df: pl.DataFrame,
    metrics: dict[str, Any],
    stats_results: dict[str, Any],
    key_metrics: dict[str, float],
    result_dir: Path,
) -> list[go.Figure]:
    """可视化图表 - 包含强度分布分析"""
    figs = []

    # 原有的可视化图
    emotion_correct_pd = metrics["emotion_accuracy"].to_pandas()
    df_pd = trials_df.to_pandas()

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "1. 情绪类型正确率对比",
            "2. 不同情绪类型的反应时分布",
            "3. 反应时与正确率的关系",
            "4. 分块正确率变化",
            "5. 中性阈值分析",
            "6. 关键指标总结",
            "7. 强度评分整体分布",
            "8. 情绪强度评分对比",
            "9. 速度-准确性权衡",
        ),
        specs=[
            [{"type": "bar"}, {"type": "box"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "table"}],
            [{"type": "histogram"}, {"type": "box"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.15,
    )

    # 图1: 不同情绪类型的识别正确率
    fig.add_trace(
        go.Bar(
            x=emotion_correct_pd["stim_type"],
            y=emotion_correct_pd["correct_rate"],
            text=[f"{rate:.1%}" for rate in emotion_correct_pd["correct_rate"]],
            textposition="auto",
            marker_color=["#636efa", "#00cc96", "#ef553b"],
            name="正确率",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(range=[0, 1.05], title_text="正确率", row=1, col=1)
    fig.update_xaxes(title_text="情绪类型", row=1, col=1)

    # 图2: 反应时分布
    for stim_type, color in zip(
        ["positive", "neutral", "negative"], ["#636efa", "#00cc96", "#ef553b"]
    ):
        rt_data = (
            trials_df.filter(pl.col("stim_type") == stim_type)["rt_clean"]
            .drop_nulls()
            .to_list()
        )
        if rt_data:
            fig.add_trace(
                go.Box(
                    y=rt_data,
                    name=stim_type,
                    marker_color=color,
                    boxmean="sd",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
    fig.update_yaxes(title_text="反应时 (秒)", row=1, col=2)
    fig.update_xaxes(title_text="情绪类型", row=1, col=2)

    # 图3: 反应时与正确率的关系
    scatter_sample = df_pd.sample(min(len(df_pd), 100), random_state=42)
    fig.add_trace(
        go.Scatter(
            x=scatter_sample["rt_clean"],
            y=scatter_sample["correct"].astype(int),
            mode="markers",
            marker=dict(
                size=8,
                color=scatter_sample["stim_type"].map(
                    {"positive": 0, "neutral": 1, "negative": 2}
                ),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title=dict(text="情绪类型", side="top"),
                    tickvals=[0, 1, 2],
                    ticktext=["积极", "中性", "消极"],
                    len=0.1,
                    y=0.5,
                    x=1.1,
                    thickness=15,
                    orientation="h",
                ),
            ),
            text=scatter_sample["stim_type"],
            hovertemplate="<b>反应时</b>: %{x:.3f}s<br><b>是否正确</b>: %{y}<br><b>类型</b>: %{text}<extra></extra>",
            name="RT vs 正确率",
        ),
        row=1,
        col=3,
    )
    fig.update_yaxes(
        tickvals=[0, 1], ticktext=["错误", "正确"], title_text="是否正确", row=1, col=3
    )
    fig.update_xaxes(title_text="反应时 (秒)", row=1, col=3)

    # 图4: 分块正确率变化
    block_correct_pd = metrics["block_accuracy"].to_pandas()
    fig.add_trace(
        go.Scatter(
            x=block_correct_pd["block_index"],
            y=block_correct_pd["correct_rate"],
            mode="lines+markers",
            marker=dict(size=12),
            line=dict(width=3),
            hovertemplate="<b>区块</b>: %{x}<br><b>正确率</b>: %{y:.1%}<extra></extra>",
            name="分块正确率",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(range=[0, 1.05], title_text="正确率", row=2, col=1)
    fig.update_xaxes(title_text="区块编号", row=2, col=1)

    # 图5: 中性阈值分析
    if "neutral_threshold_by_stimulus" in metrics:
        neutral_threshold_data = metrics["neutral_threshold_by_stimulus"]
        for stim_type in ["positive", "negative"]:
            stim_data = neutral_threshold_data.filter(
                pl.col("stim_type") == stim_type
            ).to_pandas()
            if not stim_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=stim_data["stim_type"],
                        y=stim_data["count"],
                        name=f"{stim_type}被判断为中性",
                        text=stim_data["count"],
                        textposition="auto",
                        marker_color="lightblue",
                    ),
                    row=2,
                    col=2,
                )
    fig.update_yaxes(title_text="被判断为中性的次数", row=2, col=2)
    fig.update_xaxes(title_text="刺激类型", row=2, col=2)

    # 图6: 关键指标总结表格
    table_metrics = []
    table_values = []
    table_descriptions = []

    for metric, name in zip(key_metrics, metric_names):
        if metric in key_metrics:
            table_metrics.append(name)
            value = key_metrics[metric]
            if "accuracy" in metric:
                table_values.append(f"{value:.3f}")
                table_descriptions.append("正确率指标")
            elif "rt" in metric:
                table_values.append(f"{value:.3f}s")
                table_descriptions.append("反应时指标")
            else:
                table_values.append(f"{value:.3f}")
                table_descriptions.append("其他指标")

    metrics_table = go.Table(
        header=dict(
            values=["指标", "值", "解释"], fill_color="lightblue", align="left"
        ),
        cells=dict(
            values=[table_metrics, table_values, table_descriptions],
            fill_color="lavender",
            align="left",
        ),
    )
    fig.add_trace(metrics_table, row=2, col=3)

    # 图7: 强度评分整体分布
    if metrics.get("has_intensity_data", False):
        intensity_values = (
            trials_df.filter(pl.col("stim_type") != "neutral")["intensity"]
            .drop_nulls()
            .to_list()
        )
        if intensity_values:
            fig.add_trace(
                go.Histogram(
                    x=intensity_values,
                    nbinsx=20,
                    name="强度评分分布",
                    marker_color="lightblue",
                    opacity=0.7,
                ),
                row=3,
                col=1,
            )
    fig.update_xaxes(title_text="强度评分", range=[0, 10], row=3, col=1)
    fig.update_yaxes(title_text="频数", row=3, col=1)

    # 图8: 情绪强度评分对比
    if metrics.get("has_intensity_data", False):
        for stim_type in ["positive", "negative"]:
            stim_data = (
                trials_df.filter(
                    (pl.col("stim_type") == stim_type)
                    & (pl.col("intensity").is_not_null())
                )["intensity"]
                .drop_nulls()
                .to_list()
            )
            if stim_data:
                fig.add_trace(
                    go.Box(
                        y=stim_data,
                        name=stim_type,
                        boxpoints="outliers",
                        marker_color="#00cc96"
                        if stim_type == "positive"
                        else "#ef553b",
                        showlegend=False,
                    ),
                    row=3,
                    col=2,
                )
    fig.update_yaxes(title_text="强度评分", range=[0, 10], row=3, col=2)
    fig.update_xaxes(title_text="情绪类型", row=3, col=2)

    # 图9: 速度-准确性权衡
    if "speed_accuracy_tradeoff" in stats_results:
        speed_data = stats_results["speed_accuracy_tradeoff"]
        fig.add_trace(
            go.Scatter(
                x=speed_data["rt_quartile"],
                y=speed_data["correct"],
                mode="lines+markers",
                name="速度-准确性权衡",
                line=dict(width=3, color="purple"),
                marker=dict(size=12),
            ),
            row=3,
            col=3,
        )
        fig.update_yaxes(range=[0, 1.05], title_text="正确率", row=3, col=3)
        fig.update_xaxes(title_text="反应时四分位数", row=3, col=3)

    fig.update_layout(
        height=1400,
        width=1600,
        title=dict(
            text="面部情绪识别实验行为数据分析",
            font=dict(size=20, family="Arial Black"),
            x=0.5,
        ),
        showlegend=True,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    html_path = result_dir / "emotion_face_visualization.html"
    fig.write_html(str(html_path))

    figs.append(fig)

    # 添加强度分布详细分析图
    if metrics.get("has_intensity_data", False):
        intensity_fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "强度评分分布直方图",
                "情绪类型强度对比",
                "评分与标签强度一致性",
                "强度差异分析",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "box"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # 准备强度数据
        intensity_trials = trials_df.filter(pl.col("stim_type") != "neutral")
        intensity_pd = intensity_trials.to_pandas()

        # 图1: 强度评分分布直方图
        intensity_fig.add_trace(
            go.Histogram(
                x=intensity_pd["intensity"],
                nbinsx=20,
                name="强度评分分布",
                marker_color="lightblue",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )
        intensity_fig.update_xaxes(title_text="强度评分", range=[0, 10], row=1, col=1)
        intensity_fig.update_yaxes(title_text="频数", row=1, col=1)

        # 图2: 情绪类型强度对比箱线图
        for stim_type in ["positive", "negative"]:
            stim_data = intensity_pd[intensity_pd["stim_type"] == stim_type][
                "intensity"
            ]
            if len(stim_data) > 0:
                intensity_fig.add_trace(
                    go.Box(
                        y=stim_data,
                        name=stim_type,
                        boxpoints="outliers",
                        marker_color="#00cc96"
                        if stim_type == "positive"
                        else "#ef553b",
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
        intensity_fig.update_yaxes(title_text="强度评分", range=[0, 10], row=1, col=2)
        intensity_fig.update_xaxes(title_text="情绪类型", row=1, col=2)

        # 图3: 评分与标签强度一致性散点图
        if len(intensity_pd) > 0:
            intensity_fig.add_trace(
                go.Scatter(
                    x=intensity_pd["label_intensity"],
                    y=intensity_pd["intensity"],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=intensity_pd["stim_type"].map(
                            {"positive": 0, "negative": 1}
                        ),
                        colorscale=["#00cc96", "#ef553b"],
                        showscale=True,
                        colorbar=dict(
                            title="情绪类型",
                            tickvals=[0, 1],
                            ticktext=["积极", "消极"],
                            len=0.4,
                            y=0.3,
                        ),
                    ),
                    name="评分 vs 标签",
                    text=intensity_pd["stim_type"],
                    hovertemplate="<b>标签强度</b>: %{x}<br><b>被试评分</b>: %{y}<br><b>情绪类型</b>: %{text}<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # 添加对角线
            max_val = max(intensity_pd[["label_intensity", "intensity"]].max().max(), 9)
            intensity_fig.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode="lines",
                    line=dict(dash="dash", color="gray", width=2),
                    name="理想一致性线",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            # 添加回归线
            if len(intensity_pd) >= 2:
                z = np.polyfit(
                    intensity_pd["label_intensity"], intensity_pd["intensity"], 1
                )
                p = np.poly1d(z)
                x_range = np.linspace(0, max_val, 100)
                intensity_fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=p(x_range),
                        mode="lines",
                        line=dict(color="red", width=2),
                        name="回归线",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

        intensity_fig.update_xaxes(title_text="标签强度", range=[0, 10], row=2, col=1)
        intensity_fig.update_yaxes(title_text="被试评分", range=[0, 10], row=2, col=1)

        # 图4: 强度差异分析条形图
        if "intensity_diff_by_emotion" in metrics:
            diff_data = metrics["intensity_diff_by_emotion"].to_pandas()

            for i, row in diff_data.iterrows():
                intensity_fig.add_trace(
                    go.Bar(
                        x=[row["stim_type"]],
                        y=[row["mean_diff"]],
                        name=f"{row['stim_type']}差异",
                        text=f"{row['mean_diff']:.2f}",
                        textposition="auto",
                        marker_color="#00cc96"
                        if row["stim_type"] == "positive"
                        else "#ef553b",
                        error_y=dict(
                            type="data", array=[row["std_diff"]], visible=True
                        ),
                        showlegend=False,
                    ),
                    row=2,
                    col=2,
                )

            # 添加零线
            intensity_fig.add_hline(
                y=0, line_dash="dash", line_color="black", row=2, col=2
            )

        intensity_fig.update_yaxes(title_text="评分差异均值", row=2, col=2)
        intensity_fig.update_xaxes(title_text="情绪类型", row=2, col=2)

        intensity_fig.update_layout(
            height=800,
            width=1000,
            title=dict(
                text="面部情绪强度评分分布分析",
                font=dict(size=20, family="Arial Black"),
                x=0.5,
            ),
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        intensity_path = result_dir / "intensity_distribution_analysis.html"
        intensity_fig.write_html(str(intensity_path))
        figs.append(intensity_fig)

    if "neutral_threshold_by_stimulus" in metrics:
        create_neutral_threshold_visualization(metrics, result_dir)

    return figs


def create_neutral_threshold_visualization(
    metrics: dict[str, Any], result_dir: Path
) -> Path:
    """中性阈值专用可视化"""
    if "neutral_threshold_by_stimulus" not in metrics:
        return None

    threshold_data = metrics["neutral_threshold_by_stimulus"].to_pandas()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=threshold_data["stim_type"],
            y=threshold_data["count"],
            name="被判断为中性的次数",
            text=threshold_data["count"],
            textposition="auto",
            marker_color="lightblue",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=threshold_data["stim_type"],
            y=threshold_data["mean_label_intensity"],
            mode="lines+markers",
            name="平均标签强度",
            yaxis="y2",
            line=dict(color="red", width=2),
            marker=dict(size=10, symbol="diamond"),
        )
    )

    fig.update_layout(
        title=dict(
            text="中性阈值分析：不同刺激类型被判断为中性情况",
            font=dict(size=16, family="Arial"),
            x=0.5,
        ),
        yaxis=dict(title="被判断为中性的次数"),
        yaxis2=dict(
            title="平均标签强度",
            overlaying="y",
            side="right",
            range=[0, 10],
        ),
        hovermode="x unified",
        template="plotly_white",
    )

    threshold_path = result_dir / "neutral_threshold_analysis.html"
    fig.write_html(str(threshold_path))

    return threshold_path


def save_results(
    metrics: dict[str, Any],
    stats_results: dict[str, Any],
    key_metrics: dict[str, float],
    result_dir: Path,
) -> dict[str, Path]:
    """保存结果"""
    saved_files = {}

    # 计算全面的指标
    comprehensive_metrics = calculate_comprehensive_key_metrics(None, metrics)

    # 保存全面指标
    comprehensive_df = pd.DataFrame([comprehensive_metrics])
    comprehensive_path = result_dir / "emotion_face_comprehensive_metrics.csv"
    comprehensive_df.to_csv(comprehensive_path, index=False)
    saved_files["comprehensive_metrics"] = comprehensive_path

    key_metrics_df = pd.DataFrame([key_metrics])
    key_metrics_path = result_dir / "emotion_face_key_metrics.csv"
    key_metrics_df.to_csv(key_metrics_path, index=False)
    saved_files["key_metrics"] = key_metrics_path

    basic_metrics = {
        "overall_accuracy": metrics["overall_accuracy"],
        "median_rt": metrics["median_rt"],
        "total_trials": metrics["total_trials"],
    }
    basic_metrics_df = pd.DataFrame([basic_metrics])
    basic_metrics_path = result_dir / "emotion_face_basic_metrics.csv"
    basic_metrics_df.to_csv(basic_metrics_path, index=False)
    saved_files["basic_metrics"] = basic_metrics_path

    emotion_acc_df = metrics["emotion_accuracy"].to_pandas()
    emotion_acc_path = result_dir / "emotion_face_accuracy_by_type.csv"
    emotion_acc_df.to_csv(emotion_acc_path, index=False)
    saved_files["accuracy_by_type"] = emotion_acc_path

    rt_df = metrics["reaction_time"].to_pandas()
    rt_path = result_dir / "emotion_face_reaction_time.csv"
    rt_df.to_csv(rt_path, index=False)
    saved_files["reaction_time"] = rt_path

    block_acc_df = metrics["block_accuracy"].to_pandas()
    block_acc_path = result_dir / "emotion_face_block_accuracy.csv"
    block_acc_df.to_csv(block_acc_path, index=False)
    saved_files["block_accuracy"] = block_acc_path

    if stats_results:
        stats_df = pd.DataFrame()
        for test_name, test_result in stats_results.items():
            if isinstance(test_result, dict):
                test_df = pd.DataFrame([test_result])
                test_df["test"] = test_name
                stats_df = pd.concat([stats_df, test_df], ignore_index=True)

        if not stats_df.empty:
            stats_path = result_dir / "emotion_face_statistical_tests.csv"
            stats_df.to_csv(stats_path, index=False)
            saved_files["statistical_tests"] = stats_path

    if "intensity_by_emotion" in metrics:
        intensity_df = metrics["intensity_by_emotion"].to_pandas()
        intensity_path = result_dir / "emotion_face_intensity_by_emotion.csv"
        intensity_df.to_csv(intensity_path, index=False)
        saved_files["intensity_by_emotion"] = intensity_path

    if "intensity_correlation" in metrics:
        corr_df = pd.DataFrame([metrics["intensity_correlation"]])
        corr_path = result_dir / "emotion_face_intensity_correlation.csv"
        corr_df.to_csv(corr_path, index=False)
        saved_files["intensity_correlation"] = corr_path

    if "neutral_threshold_by_stimulus" in metrics:
        threshold_df = metrics["neutral_threshold_by_stimulus"].to_pandas()
        threshold_path = result_dir / "emotion_face_neutral_threshold.csv"
        threshold_df.to_csv(threshold_path, index=False)
        saved_files["neutral_threshold"] = threshold_path

    # 保存强度分布数据
    if "intensity_distribution" in metrics:
        intensity_dist_df = pd.DataFrame([metrics["intensity_distribution"]])
        intensity_dist_path = result_dir / "emotion_face_intensity_distribution.csv"
        intensity_dist_df.to_csv(intensity_dist_path, index=False)
        saved_files["intensity_distribution"] = intensity_dist_path

    if "intensity_difference" in metrics:
        intensity_diff_df = pd.DataFrame([metrics["intensity_difference"]])
        intensity_diff_path = result_dir / "emotion_face_intensity_difference.csv"
        intensity_diff_df.to_csv(intensity_diff_path, index=False)
        saved_files["intensity_difference"] = intensity_diff_path

    if "intensity_diff_by_emotion" in metrics:
        intensity_diff_emotion_df = metrics["intensity_diff_by_emotion"].to_pandas()
        intensity_diff_emotion_path = (
            result_dir / "emotion_face_intensity_diff_by_emotion.csv"
        )
        intensity_diff_emotion_df.to_csv(intensity_diff_emotion_path, index=False)
        saved_files["intensity_diff_by_emotion"] = intensity_diff_emotion_path

    return saved_files


def analyze_emotion_face_data(
    df: pl.DataFrame,
    target_blocks: list[int] = [0, 1],
    result_dir: Path = Path("results"),
) -> dict[str, Any]:
    """分析单个被试的面部情绪识别数据"""
    # 1. 加载和预处理数据
    trials_df = load_and_preprocess_data(df)

    # 2. 计算基本指标
    metrics = calculate_basic_metrics(trials_df)

    # 3. 计算强度指标和中性阈值
    intensity_metrics = calculate_intensity_metrics(trials_df)
    metrics.update(intensity_metrics)

    # 4. 执行统计检验
    stats_results = perform_statistical_tests(trials_df)

    # 5. 计算关键指标
    key_metrics_dict = calculate_key_metrics(trials_df, metrics)

    # 6. 计算全面指标
    comprehensive_metrics = calculate_comprehensive_key_metrics(trials_df, metrics)

    # 7. 创建可视化
    figs = create_visualizations(  # noqa: F841
        trials_df, metrics, stats_results, key_metrics_dict, result_dir
    )

    # 8. 保存结果
    saved_files = save_results(metrics, stats_results, key_metrics_dict, result_dir)

    # 9. 生成报告
    report = {
        "data_summary": {
            "total_trials": trials_df.height,
            "num_blocks": len(trials_df["block_index"].unique()),
            "overall_accuracy": float(metrics["overall_accuracy"]),
            "median_rt": float(metrics["median_rt"]),
        },
        "key_metrics": key_metrics_dict,  # 精简指标
        "comprehensive_metrics": comprehensive_metrics,  # 全面指标
        "metrics": metrics,
        "statistical_results": stats_results,
        "saved_files": saved_files,
        "trials_df": trials_df,  # 保存原始数据供组分析使用
    }

    print(f"\n✅ 分析完成！结果保存在: {result_dir}")
    return report


def create_group_comparison_visualizations_single_group(
    group_metrics: list[dict[str, float]],
    group_trials: list[pl.DataFrame] = None,
    group_name: str = None,
    result_dir: Path = None,
) -> list[go.Figure]:
    """单个组的组分析可视化 - 使用精简指标"""
    figs = []

    # 原有的组分析可视化
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "1. 各被试总体正确率分布",
            "2. 情绪类型正确率对比",
            "3. 反应时指标对比",
            "4. 描述性统计",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "box"}, {"type": "table"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # 图1: 各被试总体正确率分布
    accuracy_values = [m.get("overall_accuracy", 0) for m in group_metrics]
    subjects = [f"被试{i + 1}" for i in range(len(group_metrics))]

    fig.add_trace(
        go.Bar(
            x=subjects,
            y=accuracy_values,
            name="总体正确率",
            marker_color="lightblue",
            text=[f"{v:.3f}" for v in accuracy_values],
            textposition="auto",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(range=[0, 1.05], title_text="正确率", row=1, col=1)
    fig.update_xaxes(title_text="被试", row=1, col=1)

    # 图2: 情绪类型正确率对比
    emotion_acc_values = {"positive": [], "negative": [], "neutral": []}
    for m in group_metrics:
        for emotion in emotion_acc_values.keys():
            key = f"{emotion}_accuracy"
            emotion_acc_values[emotion].append(m.get(key, 0))

    for emotion, color in zip(
        ["positive", "negative", "neutral"], ["#00cc96", "#ef553b", "#636efa"]
    ):
        if any(v > 0 for v in emotion_acc_values[emotion]):
            fig.add_trace(
                go.Box(
                    y=emotion_acc_values[emotion],
                    name=emotion,
                    boxpoints="all",
                    marker_color=color,
                    showlegend=True,
                ),
                row=1,
                col=2,
            )
    fig.update_yaxes(range=[0, 1.05], title_text="正确率", row=1, col=2)
    fig.update_xaxes(title_text="情绪类型", row=1, col=2)

    # 图3: 反应时指标对比
    rt_metrics = [m.get("median_rt", 0) for m in group_metrics]
    rt_pos_neg_diff = [m.get("positive_neg_rt_diff", 0) for m in group_metrics]

    fig.add_trace(
        go.Box(
            y=rt_metrics,
            name="中位反应时",
            boxpoints="all",
            marker_color="orange",
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Box(
            y=rt_pos_neg_diff,
            name="积极-消极反应时差",
            boxpoints="all",
            marker_color="purple",
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="反应时(秒)", row=2, col=1)
    fig.update_xaxes(title_text="指标", row=2, col=1)

    # 图4: 描述性统计表格
    metrics_df = pd.DataFrame(group_metrics)
    descriptive_stats = []

    for metric, name in zip(key_metrics, metric_names):
        if metric in metrics_df.columns:
            values = metrics_df[metric].dropna()
            if len(values) > 0:
                # 跳过非数值列
                if pd.api.types.is_numeric_dtype(values):
                    descriptive_stats.append(
                        [
                            name,
                            f"{values.mean():.3f}",
                            f"{values.std():.3f}",
                            f"{values.min():.3f}",
                            f"{values.max():.3f}",
                            f"{len(values)}",
                        ]
                    )

    if descriptive_stats:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["指标", "均值", "标准差", "最小值", "最大值", "样本数"],
                    fill_color="lightgreen",
                    align="left",
                    font=dict(size=10),
                ),
                cells=dict(
                    values=np.array(descriptive_stats).T,
                    fill_color="honeydew",
                    align="left",
                    font=dict(size=9),
                ),
                columnwidth=[0.2, 0.15, 0.15, 0.15, 0.15, 0.15],
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=800,
        width=1200,
        title=dict(
            text=f"面部情绪识别{group_name}组分析报告"
            if group_name
            else "面部情绪识别组分析报告",
            font=dict(size=22, family="Arial Black"),
            x=0.5,
        ),
        showlegend=True,
        template="plotly_white",
    )

    figs.append(fig)

    # 如果提供了trials数据，添加强度分布分析
    if group_trials and len(group_trials) > 0:
        # 合并所有被试的强度数据
        all_intensity_data = []
        for i, trials_df in enumerate(group_trials):
            intensity_trials = trials_df.filter(
                (pl.col("stim_type") != "neutral") & (pl.col("intensity").is_not_null())
            )

            if intensity_trials.height > 0:
                df = intensity_trials.to_pandas()
                df["subject_id"] = f"被试{i + 1}"
                all_intensity_data.append(df)

        if all_intensity_data:
            combined_df = pd.concat(all_intensity_data, ignore_index=True)

            # 创建组内强度分布分析图
            intensity_fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "组内强度评分分布",
                    "被试间强度对比",
                    "情绪类型强度分布",
                    "强度评分一致性分析",
                ),
                specs=[
                    [{"type": "histogram"}, {"type": "box"}],
                    [{"type": "box"}, {"type": "scatter"}],
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.1,
            )

            # 图1: 组内强度评分分布直方图
            intensity_fig.add_trace(
                go.Histogram(
                    x=combined_df["intensity"],
                    nbinsx=20,
                    name="组强度分布",
                    marker_color="lightblue",
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )
            intensity_fig.update_xaxes(
                title_text="强度评分", range=[0, 10], row=1, col=1
            )
            intensity_fig.update_yaxes(title_text="频数", row=1, col=1)

            # 图2: 被试间强度对比箱线图
            subject_ids = combined_df["subject_id"].unique()
            for subject_id in subject_ids:
                subject_data = combined_df[combined_df["subject_id"] == subject_id][
                    "intensity"
                ]
                if len(subject_data) > 0:
                    intensity_fig.add_trace(
                        go.Box(
                            y=subject_data,
                            name=subject_id,
                            boxpoints="outliers",
                            marker_color="lightblue",
                            showlegend=False,
                        ),
                        row=1,
                        col=2,
                    )
            intensity_fig.update_yaxes(
                title_text="强度评分", range=[0, 10], row=1, col=2
            )
            intensity_fig.update_xaxes(title_text="被试", row=1, col=2)

            # 图3: 情绪类型强度分布
            for stim_type in ["positive", "negative"]:
                stim_data = combined_df[combined_df["stim_type"] == stim_type][
                    "intensity"
                ]
                if len(stim_data) > 0:
                    intensity_fig.add_trace(
                        go.Box(
                            y=stim_data,
                            name=stim_type,
                            boxpoints="outliers",
                            marker_color="#00cc96"
                            if stim_type == "positive"
                            else "#ef553b",
                            showlegend=True,
                        ),
                        row=2,
                        col=1,
                    )
            intensity_fig.update_yaxes(
                title_text="强度评分", range=[0, 10], row=2, col=1
            )
            intensity_fig.update_xaxes(title_text="情绪类型", row=2, col=1)

            # 图4: 强度评分一致性分析
            if len(combined_df) > 0:
                # 计算每个标签强度的平均评分
                label_stats = (
                    combined_df.groupby("label_intensity")
                    .agg({"intensity": ["mean", "std", "count"]})
                    .reset_index()
                )
                label_stats.columns = [
                    "label_intensity",
                    "mean_intensity",
                    "std_intensity",
                    "count",
                ]

                intensity_fig.add_trace(
                    go.Scatter(
                        x=label_stats["label_intensity"],
                        y=label_stats["mean_intensity"],
                        error_y=dict(
                            type="data",
                            array=label_stats["std_intensity"],
                            visible=True,
                        ),
                        mode="lines+markers",
                        name="平均评分",
                        line=dict(width=3, color="blue"),
                        marker=dict(size=10),
                        text=[f"n={n}" for n in label_stats["count"]],
                        hovertemplate="<b>标签强度</b>: %{x}<br><b>平均评分</b>: %{y:.2f}<br><b>标准差</b>: %{error_y.array}<br><b>试次数</b>: %{text}<extra></extra>",
                    ),
                    row=2,
                    col=2,
                )

                # 添加对角线
                max_val = max(
                    label_stats["label_intensity"].max(),
                    label_stats["mean_intensity"].max(),
                    9,
                )
                intensity_fig.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode="lines",
                        line=dict(dash="dash", color="gray", width=2),
                        name="理想一致性线",
                        showlegend=True,
                    ),
                    row=2,
                    col=2,
                )

            intensity_fig.update_xaxes(
                title_text="标签强度", range=[0, 10], row=2, col=2
            )
            intensity_fig.update_yaxes(
                title_text="平均评分", range=[0, 10], row=2, col=2
            )

            intensity_fig.update_layout(
                height=800,
                width=1000,
                title=dict(
                    text=f"{group_name}组强度评分分布分析"
                    if group_name
                    else "组强度评分分布分析",
                    font=dict(size=20, family="Arial Black"),
                    x=0.5,
                ),
                showlegend=True,
                template="plotly_white",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            figs.append(intensity_fig)

    return figs


def create_group_comparison_visualizations(
    control_metrics: list[dict[str, float]],
    experimental_metrics: list[dict[str, float]],
    control_name: str = "对照组",
    experimental_name: str = "实验组",
) -> list[go.Figure]:
    """创建组间比较可视化 - 使用全部指标"""
    figs = []

    # 将所有指标分组显示（每6个指标一组）
    n_metrics_per_group = 6
    n_groups = (len(all_metrics) + n_metrics_per_group - 1) // n_metrics_per_group

    for group_idx in range(n_groups):
        start_idx = group_idx * n_metrics_per_group
        end_idx = min(start_idx + n_metrics_per_group, len(all_metrics))

        current_metrics = all_metrics[start_idx:end_idx]
        current_names = all_metric_names[start_idx:end_idx]

        # 计算每个子图的行列数
        n_plots = len(current_metrics)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=current_names,
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        for idx, (metric, name) in enumerate(zip(current_metrics, current_names)):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1

            # 收集两组数据
            control_values = []
            experimental_values = []

            for m in control_metrics:
                if metric in m and not np.isnan(m[metric]):
                    control_values.append(m[metric])

            for m in experimental_metrics:
                if metric in m and not np.isnan(m[metric]):
                    experimental_values.append(m[metric])

            # 创建箱线图
            if control_values:
                fig.add_trace(
                    go.Box(
                        y=control_values,
                        name=control_name,
                        boxpoints="outliers",
                        marker_color="lightgreen",
                        showlegend=(idx == 0),
                        jitter=0.3,
                        pointpos=-1.8,
                    ),
                    row=row,
                    col=col,
                )

            if experimental_values:
                fig.add_trace(
                    go.Box(
                        y=experimental_values,
                        name=experimental_name,
                        boxpoints="outliers",
                        marker_color="lightcoral",
                        showlegend=(idx == 0),
                        jitter=0.3,
                        pointpos=-1.8,
                    ),
                    row=row,
                    col=col,
                )

            # 设置y轴标签
            # if "accuracy" in metric or "正确率" in name:
            #     fig.update_yaxes(range=[0, 1.05], row=row, col=col)
            if "rt" in metric or "反应时" in name:
                fig.update_yaxes(title_text="反应时(秒)", row=row, col=col)
            # elif "intensity" in metric or "强度" in name:
            #     fig.update_yaxes(range=[0, 10], row=row, col=col)

        fig.update_layout(
            height=400 * n_rows,
            width=800,
            margin=dict(l=50, r=50, t=50, b=50),  # 均匀边距
            title=dict(
                text=f"{control_name} vs {experimental_name} 指标比较 (第{group_idx + 1}/{n_groups}组)",
                font=dict(size=20, family="Arial Black"),
                x=0.5,
            ),
            showlegend=True,
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        figs.append(fig)

    return figs


def run_single_emotion_analysis(file_path: Path, result_dir: Path = None):
    """单个被试的面部情绪识别分析"""
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    if result_dir is None:
        result_dir = file_path.parent / "emotion_face_results"

    result_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(file_path)

    result = analyze_emotion_face_data(
        df=df, target_blocks=[0, 1], result_dir=result_dir
    )

    print(f"\n✅ 分析完成！结果保存在: {result_dir}")
    return result


def run_group_emotion_analysis(
    data_files: list[Path],
    result_dir: Path = None,
    group_name: str = None,
):
    """组面部情绪识别分析 - 使用精简指标"""
    if result_dir is None:
        result_dir = Path("emotion_face_group_results")

    if group_name is not None:
        result_dir = result_dir / group_name

    result_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    group_metrics = []  # 精简指标
    comprehensive_metrics = []  # 全面指标
    all_trials = []  # 保存所有被试的trials_df

    for i, file_path in enumerate(data_files):
        print(f"分析被试 {i + 1}/{len(data_files)}: {file_path.name}")

        try:
            df = pl.read_csv(file_path)
            subject_id = file_path.stem.split("-")[0]

            subject_result_dir = result_dir / subject_id
            subject_result_dir.mkdir(parents=True, exist_ok=True)

            result = analyze_emotion_face_data(
                df=df,
                target_blocks=[0, 1],
                result_dir=subject_result_dir,
            )
            result["subject_id"] = subject_id
            result["key_metrics"]["subject_id"] = subject_id

            if result:
                all_results.append(result)
                group_metrics.append(result["key_metrics"])  # 精简指标
                comprehensive_metrics.append(
                    result["comprehensive_metrics"]
                )  # 全面指标
                # 保存trials_df供组分析使用
                if "trials_df" in result:
                    all_trials.append(result["trials_df"])
        except Exception as e:
            print(f"❌ 被试 {file_path.name} 分析出错: {e}")

    print(f"\n共完成 {len(all_results)}/{len(data_files)} 个被试的分析")

    if len(all_results) < 2:
        print("⚠️ 被试数量不足，无法进行统计检验")
        return {
            "all_results": all_results,
            "group_metrics": group_metrics,
            "comprehensive_metrics": comprehensive_metrics,
            "all_trials": all_trials,
        }

    # 统计检验 - 使用精简指标
    statistical_results = {}

    for metric in key_metrics:
        group_values = [m.get(metric, np.nan) for m in group_metrics if metric in m]
        group_values = [v for v in group_values if not np.isnan(v)]

        if len(group_values) < 2:
            statistical_results[metric] = {"error": "样本量不足"}
            continue

        # 单样本t检验
        if "accuracy" in metric or metric == "overall_accuracy":
            ref_value = 0.8
        elif "rt" in metric:
            ref_value = 0.5
        else:
            ref_value = 0

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

        sample_size_info = {}
        if cohens_d is not None and cohens_d != 0:
            try:
                sample_size_info = calculate_sample_size(
                    effect_size=abs(cohens_d),
                    alpha=0.05,
                    power=0.8,
                    test_type="one_sample",
                )
            except (OverflowError, ZeroDivisionError):
                sample_size_info = {}

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

    # 保存指标数据
    all_metrics_df = pd.DataFrame([r["key_metrics"] for r in all_results])
    all_metrics_df.to_csv(result_dir / "group_key_metrics.csv", index=False)

    comprehensive_metrics_df = pd.DataFrame(comprehensive_metrics)
    comprehensive_metrics_df.to_csv(
        result_dir / "group_comprehensive_metrics.csv", index=False
    )

    group_mean_metrics = all_metrics_df.mean(numeric_only=True).to_dict()
    group_std_metrics = all_metrics_df.std(numeric_only=True).to_dict()

    stats_df = pd.DataFrame(
        [group_mean_metrics, group_std_metrics], index=["mean", "std"]
    ).T
    stats_df.to_csv(result_dir / "group_statistics.csv")

    sample_size_data = []
    for metric, result in statistical_results.items():
        if (
            "required_sample_size_per_group" in result
            and result["required_sample_size_per_group"] is not None
        ):
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

    if statistical_results:
        stats_test_df = pd.DataFrame(statistical_results).T
        stats_test_df.to_csv(result_dir / "group_statistical_tests.csv")

    fig_spec = create_group_comparison_visualizations_single_group(
        group_metrics=group_metrics,
        group_trials=all_trials,
        group_name=group_name,
        result_dir=result_dir,
    )

    fig_common = create_common_single_group_figures(
        group_metrics, statistical_results, key_metrics, metric_names
    )

    figs = fig_spec + fig_common
    save_html_report(
        save_dir=result_dir,
        save_name=f"emotion_face-{group_name}_group-analysis_report"
        if group_name
        else "emotion_face_group-analysis_report",
        figures=figs,
        title=f"面部情绪识别{group_name}组分析" if group_name else "面部情绪识别组分析",
    )

    return {
        "all_results": all_results,
        "group_metrics": group_metrics,
        "comprehensive_metrics": comprehensive_metrics,
        "statistical_results": statistical_results,
        "group_mean": group_mean_metrics,
        "group_std": group_std_metrics,
        "all_trials": all_trials,
    }


def run_groups_emotion_analysis(
    control_files: list[Path],
    experimental_files: list[Path],
    result_dir: Path = Path("emotion_face_group_comparison_results"),
    groups: list[str] = None,
) -> dict[str, Any]:
    """比较对照组和实验组 - 使用全面指标进行比较"""

    control_name = groups[0] if groups else "control"

    control_group_results = run_group_emotion_analysis(
        control_files, result_dir, control_name
    )
    control_results = control_group_results["all_results"]
    control_comprehensive_metrics = control_group_results["comprehensive_metrics"]

    experimental_name = groups[1] if groups and len(groups) > 1 else "experimental"

    experimental_group_results = run_group_emotion_analysis(
        experimental_files, result_dir, experimental_name
    )
    experimental_results = experimental_group_results["all_results"]
    experimental_comprehensive_metrics = experimental_group_results[
        "comprehensive_metrics"
    ]

    if len(control_results) < 2 or len(experimental_results) < 2:
        print("⚠️ 任一组被试数量不足，无法进行组间统计检验")
        return {
            "control_results": control_results,
            "experimental_results": experimental_results,
            "control_comprehensive_metrics": control_comprehensive_metrics,
            "experimental_comprehensive_metrics": experimental_comprehensive_metrics,
        }

    print("\n检查正态性和方差齐性...")
    # 使用全面指标检查正态性和方差齐性
    normality_results = check_normality_and_homoscedasticity(
        control_comprehensive_metrics + experimental_comprehensive_metrics, all_metrics
    )

    print("\n执行组间比较分析...")
    # 使用全面指标进行组间比较
    comparison_results = perform_group_comparisons(
        control_comprehensive_metrics, experimental_comprehensive_metrics, all_metrics
    )

    print("\n保存组分析结果...")

    # 保存全面指标数据
    all_control_metrics_df = pd.DataFrame(control_comprehensive_metrics)
    all_control_metrics_df.insert(0, "group", control_name)

    all_experimental_metrics_df = pd.DataFrame(experimental_comprehensive_metrics)
    all_experimental_metrics_df.insert(0, "group", experimental_name)

    all_metrics_df = pd.concat(
        [all_control_metrics_df, all_experimental_metrics_df], ignore_index=True
    )
    all_metrics_df.to_csv(
        result_dir / "all_subjects_comprehensive_metrics.csv", index=False
    )

    control_stats = all_control_metrics_df.drop(columns=["group"]).describe()
    experimental_stats = all_experimental_metrics_df.drop(columns=["group"]).describe()

    control_stats.to_csv(result_dir / f"{control_name}_comprehensive_statistics.csv")
    experimental_stats.to_csv(
        result_dir / f"{experimental_name}_comprehensive_statistics.csv"
    )

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
        sample_size_df.to_csv(
            result_dir / "comprehensive_sample_size_calculations.csv", index=False
        )

    if normality_results:
        normality_df = pd.DataFrame(normality_results).T
        normality_df.to_csv(result_dir / "comprehensive_normality_tests.csv")

    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results).T
        comparison_df.to_csv(result_dir / "comprehensive_group_comparisons.csv")

    # 使用全面指标创建可视化
    fig_spec = create_group_comparison_visualizations(
        control_metrics=control_comprehensive_metrics,
        experimental_metrics=experimental_comprehensive_metrics,
        control_name=control_name,
        experimental_name=experimental_name,
    )

    # 使用全面指标创建通用比较图
    fig_common = create_common_comparison_figures(
        comparison_results, all_metrics, all_metric_names
    )

    figs = fig_spec + fig_common
    save_html_report(
        save_dir=result_dir,
        save_name=f"emotion_face-{control_name}_{experimental_name}_group-comparison_report",
        figures=figs,
        title=f"面部情绪识别{control_name}-{experimental_name}组间比较分析",
    )

    return {
        "control_results": control_results,
        "experimental_results": experimental_results,
        "control_comprehensive_metrics": control_comprehensive_metrics,
        "experimental_comprehensive_metrics": experimental_comprehensive_metrics,
        "normality_results": normality_results,
        "comparison_results": comparison_results,
    }


def run_emotion_face_analysis(cfg: DictConfig = None, data_utils: DataUtils = None):
    """面部情绪识别分析入口函数"""

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

            result_dir = result_root / "emotion_face_results"
            result_dir.mkdir(parents=True, exist_ok=True)

            result = run_single_emotion_analysis(file_path, result_dir)
            return result

        elif choice == "2":
            dir_input = input("请输入包含多个数据文件的目录路径: ").strip("'").strip()
            data_dir = Path(dir_input.strip("'").strip('"')).resolve()

            if not data_dir.exists():
                print(f"❌ 目录不存在: {data_dir}")
                return

            data_files = find_emotion_face_files(data_dir)
            print(f"找到 {len(data_files)} 个数据文件")

            result_dir = result_root / "emotion_face_group_results"
            result_dir.mkdir(parents=True, exist_ok=True)

            result = run_group_emotion_analysis(data_files, result_dir)
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

            control_files = find_emotion_face_files(control_dir)
            experimental_files = find_emotion_face_files(experimental_dir)

            print(
                f"找到 {len(control_files)} 个对照组文件和 {len(experimental_files)} 个实验组文件"
            )

            result_dir = result_root / "emotion_face_group_comparison_results"
            result_dir.mkdir(parents=True, exist_ok=True)

            result = run_groups_emotion_analysis(
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
                files = find_emotion_face_files(data_root / groups[0])
                result_dir = result_root / f"emotion_face_{groups[0]}_results"
                result_dir.mkdir(parents=True, exist_ok=True)

                run_group_emotion_analysis(files, result_dir)
            else:
                # 多个组分析
                control_files = find_emotion_face_files(data_root / groups[0])
                experimental_files = find_emotion_face_files(data_root / groups[1])
                result_dir = (
                    result_root
                    / f"emotion_face_{groups[0]}_{groups[1]}_comparison_results"
                )
                result_dir.mkdir(parents=True, exist_ok=True)
                run_groups_emotion_analysis(
                    control_files, experimental_files, result_dir, groups
                )
        else:
            # 单个被试分析
            file_path = (
                Path(cfg.output_dir)
                / data_utils.date
                / f"{data_utils.session_id}-face_recognition.csv"
            )

            if not file_path.exists():
                print(f"❌ 文件不存在: {file_path}")
                return

            result_dir = (
                result_root / str(data_utils.session_id) / "emotion_face_analysis"
            )
            result_dir.mkdir(parents=True, exist_ok=True)

            result = run_single_emotion_analysis(file_path, result_dir)
            return result


if __name__ == "__main__":
    run_emotion_face_analysis()
