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

from psycho.analysis.utils import (
    DataUtils,
    calculate_sample_size,
    check_normality_and_homoscedasticity,
    create_common_comparison_figures,
    create_common_single_group_figures,
    draw_ci_scatter,
    extract_trials_by_block,
    find_exp_files,
    perform_group_comparisons,
    save_html_report,
)

warnings.filterwarnings("ignore")

key_metrics = [
    "intensity",
    "positive_intensity",
    "negative_intensity",
    "rt",
    "positive_rt",
    "negative_rt",
    "blur_intensity",
    "blur_positive_intensity",
    "blur_negative_intensity",
    "blur_rt",
    "blur_positive_rt",
    "blur_negative_rt",
    "neutral_intensity",
    "positive_neutral_intensity",
    "negative_neutral_intensity",
]

metric_names = [
    "总体强度",
    "积极强度",
    "消极强度",
    "总体反应时",
    "积极反应时",
    "消极反应时",
    "模糊强度",
    "积极模糊强度",
    "消极模糊强度",
    "模糊反应时",
    "积极模糊反应时",
    "消极模糊反应时",
    "总体中性判定",
    "积极中性判定",
    "消极中性判定",
]

# 全局所有指标 - 多组分析使用
all_metrics = [
    # 强度指标
    "intensity",
    "positive_intensity",
    "negative_intensity",
    "blur_intensity",
    "blur_positive_intensity",
    "blur_negative_intensity",
    # 反应时指标
    "rt",
    "positive_rt",
    "negative_rt",
    "blur_rt",
    "blur_positive_rt",
    "blur_negative_rt",
    # 中性判定指标
    "neutral_intensity",
    "positive_neutral_intensity",
    "negative_neutral_intensity",
    # 强度差异和相关性指标
    "mean_intensity_diff",  # 平均强度差异=选择的强度评分－实际强度评分
    "intensity_correlation_r",  # 强度相关性r值=所选强度评分与实际强度评分的相关系数
]

all_metric_names = [
    # 强度指标
    "总体强度",
    "积极强度",
    "消极强度",
    "模糊强度",
    "积极模糊强度",
    "消极模糊强度",
    # 反应时指标
    "总体反应时",
    "积极反应时",
    "消极反应时",
    "模糊反应时",
    "积极模糊反应时",
    "消极模糊反应时",
    # 中性判定指标
    "总体中性判定",
    "积极中性判定",
    "消极中性判定",
    # 强度差异和相关性指标
    "平均强度差异",
    "强度相关性r值",
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

        # 添加强度等级划分（0-2:模糊, 3-6:中等, 7-9:清晰）
        def get_intensity_level(label_intensity):
            if label_intensity is None:
                return None
            if 0 <= label_intensity <= 2:
                return "blur"
            elif 3 <= label_intensity <= 6:
                return "mid"
            elif 7 <= label_intensity <= 9:
                return "clear"
            else:
                return None

        trials_df = trials_df.with_columns(
            pl.col("label_intensity")
            .map_elements(get_intensity_level, return_dtype=pl.Utf8, skip_nulls=False)
            .alias("intensity_level")
        )

        # 统一积极/消极的正负表示
        # 对于积极刺激，保持正数；对于消极刺激，使用正数（不再使用负数）
        trials_df = trials_df.with_columns(
            pl.when(pl.col("stim_type") == "positive")
            .then(pl.col("label_intensity"))
            .otherwise(pl.col("label_intensity"))
            .alias("label_intensity_signed")
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


def calculate_new_metrics(trials_df: pl.DataFrame) -> dict[str, Any]:
    """计算新的指标：强度、反应时、中性判定强度等"""
    metrics = {}

    # 只选择正确的试次用于强度计算
    correct_trials = trials_df.filter(pl.col("correct"))

    if correct_trials.height == 0:
        metrics["has_correct_data"] = False
        return metrics

    metrics["has_correct_data"] = True

    # 1. 各标签强度对应选择强度（平均）- 分情绪类型（使用正确试次）
    intensity_by_label_emotion = (
        correct_trials.group_by(["label_intensity_signed", "stim_type"])
        .agg(
            [
                pl.col("intensity").mean().alias("mean_intensity"),
                pl.col("intensity").std().alias("std_intensity"),
                pl.col("intensity").count().alias("n_trials"),
            ]
        )
        .sort(["label_intensity_signed", "stim_type"])
    )
    metrics["intensity_by_label_emotion"] = intensity_by_label_emotion

    # 2. 反应时指标（所有试次）- 分情绪类型
    rt_by_label_emotion = (
        trials_df.group_by(["label_intensity_signed", "stim_type"])
        .agg(
            [
                pl.col("rt_clean").mean().alias("mean_rt"),
                pl.col("rt_clean").std().alias("std_rt"),
                pl.col("rt_clean").count().alias("n_trials"),
            ]
        )
        .sort(["label_intensity_signed", "stim_type"])
    )
    metrics["rt_by_label_emotion"] = rt_by_label_emotion

    # 3. 模糊等级指标（0-2:模糊, 3-6:中等, 7-9:清晰）- 分情绪类型
    # 强度使用正确试次
    intensity_by_level_emotion = (
        correct_trials.group_by(["intensity_level", "stim_type"])
        .agg(
            [
                pl.col("intensity").mean().alias("mean_intensity"),
                pl.col("intensity").std().alias("std_intensity"),
                pl.col("intensity").count().alias("n_trials"),
            ]
        )
        .sort(["intensity_level", "stim_type"])
    )
    metrics["intensity_by_level_emotion"] = intensity_by_level_emotion

    # 反应时使用所有试次
    rt_by_level_emotion = (
        trials_df.group_by(["intensity_level", "stim_type"])
        .agg(
            [
                pl.col("rt_clean").mean().alias("mean_rt"),
                pl.col("rt_clean").std().alias("std_rt"),
                pl.col("rt_clean").count().alias("n_trials"),
            ]
        )
        .sort(["intensity_level", "stim_type"])
    )
    metrics["rt_by_level_emotion"] = rt_by_level_emotion

    # 4. 中性判定强度（使用所有试次，包括错误和正确）
    # 计算将每个强度标签判断为中性的比例，区分积极和消极
    neutral_by_label_emotion = (
        trials_df.group_by(["label_intensity_signed", "stim_type"])
        .agg(
            [
                (pl.col("choice_type") == "neutral").sum().alias("neutral_count"),
                pl.count().alias("total_count"),
            ]
        )
        .with_columns(
            [
                (pl.col("neutral_count") / pl.col("total_count")).alias(
                    "neutral_proportion"
                )
            ]
        )
        .sort(["label_intensity_signed", "stim_type"])
    )
    metrics["neutral_by_label_emotion"] = neutral_by_label_emotion

    # 5. 计算强度差异和相关性（只针对正确的试次）
    # 平均强度差异 = 选择的强度评分 - 实际强度评分
    intensity_diff_trials = correct_trials.filter(
        pl.col("intensity").is_not_null()
        & pl.col("label_intensity_signed").is_not_null()
    )

    if intensity_diff_trials.height > 0:
        # 平均强度差异
        mean_intensity_diff = intensity_diff_trials.select(
            (pl.col("intensity") - pl.col("label_intensity_signed"))
            .mean()
            .alias("mean_diff")
        )["mean_diff"][0]
        metrics["mean_intensity_diff"] = float(mean_intensity_diff)

        # 强度相关性
        intensity_corr_data = intensity_diff_trials.select(
            ["label_intensity_signed", "intensity"]
        ).to_pandas()
        if len(intensity_corr_data) >= 3:
            corr, p_val = stats.pearsonr(
                intensity_corr_data["label_intensity_signed"],
                intensity_corr_data["intensity"],
            )
            metrics["intensity_correlation_r"] = float(corr)
            metrics["intensity_correlation_p"] = float(p_val)

    # 6. 计算关键指标汇总值
    # 各标签强度对应选择强度（总体平均）- 使用正确试次
    intensity_values = correct_trials["intensity"].drop_nulls().to_list()
    if intensity_values:
        metrics["intensity"] = float(np.mean(intensity_values))

    # 按情绪类型的平均强度 - 使用正确试次
    for stim_type in ["positive", "negative"]:
        stim_data = correct_trials.filter(pl.col("stim_type") == stim_type)
        if stim_data.height > 0:
            stim_intensity = stim_data["intensity"].drop_nulls().to_list()
            if stim_intensity:
                metrics[f"{stim_type}_intensity"] = float(np.mean(stim_intensity))

    # 反应时指标 - 使用所有试次
    rt_values = trials_df["rt_clean"].drop_nulls().to_list()
    if rt_values:
        metrics["rt"] = float(np.mean(rt_values))

    for stim_type in ["positive", "negative"]:
        stim_data = trials_df.filter(pl.col("stim_type") == stim_type)
        if stim_data.height > 0:
            stim_rt = stim_data["rt_clean"].drop_nulls().to_list()
            if stim_rt:
                metrics[f"{stim_type}_rt"] = float(np.mean(stim_rt))

    # 模糊等级指标（0-2:模糊）
    # 强度使用正确试次
    blur_trials = correct_trials.filter(pl.col("intensity_level") == "blur")
    if blur_trials.height > 0:
        blur_intensity_values = blur_trials["intensity"].drop_nulls().to_list()
        if blur_intensity_values:
            metrics["blur_intensity"] = float(np.mean(blur_intensity_values))

        for stim_type in ["positive", "negative"]:
            stim_data = blur_trials.filter(pl.col("stim_type") == stim_type)
            if stim_data.height > 0:
                stim_intensity = stim_data["intensity"].drop_nulls().to_list()
                if stim_intensity:
                    metrics[f"blur_{stim_type}_intensity"] = float(
                        np.mean(stim_intensity)
                    )

    # 模糊反应时 - 使用所有试次
    blur_rt_trials = trials_df.filter(pl.col("intensity_level") == "blur")
    if blur_rt_trials.height > 0:
        blur_rt_values = blur_rt_trials["rt_clean"].drop_nulls().to_list()
        if blur_rt_values:
            metrics["blur_rt"] = float(np.mean(blur_rt_values))

        for stim_type in ["positive", "negative"]:
            stim_data = blur_rt_trials.filter(pl.col("stim_type") == stim_type)
            if stim_data.height > 0:
                stim_rt = stim_data["rt_clean"].drop_nulls().to_list()
                if stim_rt:
                    metrics[f"blur_{stim_type}_rt"] = float(np.mean(stim_rt))

    # 中性判定强度（总体平均）- 区分积极和消极
    neutral_data = metrics["neutral_by_label_emotion"]
    if neutral_data.height > 0:
        # 总体中性判定强度（积极和消极的平均）
        metrics["neutral_intensity"] = float(neutral_data["neutral_proportion"].mean())

        # 积极中性判定强度
        pos_neutral = neutral_data.filter(pl.col("stim_type") == "positive")
        if pos_neutral.height > 0:
            metrics["positive_neutral_intensity"] = float(
                pos_neutral["neutral_proportion"].mean()
            )

        # 消极中性判定强度
        neg_neutral = neutral_data.filter(pl.col("stim_type") == "negative")
        if neg_neutral.height > 0:
            metrics["negative_neutral_intensity"] = float(
                neg_neutral["neutral_proportion"].mean()
            )

    return metrics


def calculate_basic_metrics(trials_df: pl.DataFrame) -> dict[str, Any]:
    """计算基本行为指标"""
    metrics = {}

    # 计算正确率但不用于可视化
    metrics["overall_accuracy"] = trials_df["correct"].mean()
    metrics["median_rt"] = trials_df["rt_clean"].median()
    metrics["total_trials"] = trials_df.height

    # 计算新指标
    new_metrics = calculate_new_metrics(trials_df)
    metrics.update(new_metrics)

    return metrics


def calculate_comprehensive_key_metrics(
    trials_df: pl.DataFrame, metrics: dict[str, Any]
) -> dict[str, float]:
    """计算全面的关键指标"""
    key_metrics_dict = {}

    # 填充所有关键指标
    for metric in key_metrics:
        if metric in metrics:
            key_metrics_dict[metric] = metrics[metric]
        else:
            key_metrics_dict[metric] = 0.0  # 默认值

    # 添加强度差异和相关性指标
    if "mean_intensity_diff" in metrics:
        key_metrics_dict["mean_intensity_diff"] = metrics["mean_intensity_diff"]

    if "intensity_correlation_r" in metrics:
        key_metrics_dict["intensity_correlation_r"] = metrics["intensity_correlation_r"]

    # 处理NaN值
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


def create_single_visualizations(
    trials_df: pl.DataFrame,
    metrics: dict[str, Any],
    key_metrics: dict[str, float],
    result_dir: Path,
) -> list[go.Figure]:
    """单人可视化图表"""
    figs = []

    # 创建主可视化图 - 3行3列布局
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "各标签强度对应选择强度",
            "各标签强度对应反应时",
            "模糊等级对应选择强度",
            "模糊等级对应反应时",
            "中性判定强度",
            "强度一致性分析",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # 定义等级顺序
    level_order = ["blur", "mid", "clear"]
    # 图1: 各标签强度对应选择强度（0-9，积极和消极分开）
    if "intensity_by_label_emotion" in metrics:
        intensity_data = metrics["intensity_by_label_emotion"].to_pandas()

        # 分离积极和消极数据
        pos_data = intensity_data[intensity_data["stim_type"] == "positive"]
        neg_data = intensity_data[intensity_data["stim_type"] == "negative"]

        signed_neg_data = neg_data.copy()
        signed_neg_data["label_intensity_signed"] = -signed_neg_data[
            "label_intensity_signed"
        ]
        signed_neg_data["mean_intensity"] = -signed_neg_data["mean_intensity"]
        signed_neg_data["std_intensity"] = -signed_neg_data["std_intensity"]

        # 为每个强度级别创建条形
        all_intensities = sorted(set(intensity_data["label_intensity_signed"].tolist()))

        # 存储x轴位置
        for i, intensity in enumerate(all_intensities):
            # 积极数据
            pos_intensity_data = pos_data[
                pos_data["label_intensity_signed"] == intensity
            ]
            if not pos_intensity_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=pos_intensity_data["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=pos_intensity_data["std_intensity"].values,
                            visible=True,
                        ),
                        name="积极" if intensity == all_intensities[0] else "",
                        marker_color="#00cc96",
                        legendgroup="positive",
                        showlegend=(intensity == all_intensities[0]),
                        width=0.35,
                    ),
                    row=1,
                    col=1,
                )

            # 消极数据
            neg_intensity_data = neg_data[
                neg_data["label_intensity_signed"] == intensity
            ]
            if not neg_intensity_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=neg_intensity_data["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=neg_intensity_data["std_intensity"].values,
                            visible=True,
                        ),
                        name="消极" if intensity == all_intensities[0] else "",
                        marker_color="#ef553b",
                        legendgroup="negative",
                        showlegend=(intensity == all_intensities[0]),
                        width=0.35,
                    ),
                    row=1,
                    col=1,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(all_intensities))),
            ticktext=[str(i) for i in all_intensities],
            row=1,
            col=1,
        )

        fig.update_yaxes(title_text="选择强度", range=[0, 10], row=1, col=1)
        fig.update_xaxes(title_text="标签强度", row=1, col=1)

        # 图6: 强度一致性分析（积极、消极和中性放在一张图中）
        # 处理中性刺激数据
        neutral_stim_data = trials_df.filter(pl.col("stim_type") == "neutral")
        if neutral_stim_data.height > 0:
            # 计算中性刺激的平均选择强度
            neutral_intensity = neutral_stim_data["intensity"].mean()
            neutral_std = neutral_stim_data["intensity"].std()
            neutral_n = neutral_stim_data.height

        # 中性刺激数据（如果存在）
        if (
            not pos_data.empty
            and not signed_neg_data.empty
            and "neutral_intensity" in locals()
            and neutral_n > 0
        ):
            pos_data = pos_data.sort_values("label_intensity_signed")
            pos_upper = pos_data["mean_intensity"] + pos_data["std_intensity"]
            pos_lower = pos_data["mean_intensity"] - pos_data["std_intensity"]
            pos_main = pos_data["mean_intensity"]

            signed_neg_data = signed_neg_data.sort_values("label_intensity_signed")
            signed_neg_upper = (
                signed_neg_data["mean_intensity"] + signed_neg_data["std_intensity"]
            )
            signed_neg_lower = (
                signed_neg_data["mean_intensity"] - signed_neg_data["std_intensity"]
            )
            signed_neg_main = signed_neg_data["mean_intensity"]

            x_raw = (
                pos_data["label_intensity_signed"].tolist()
                + signed_neg_data["label_intensity_signed"].tolist()
                + [0]
            )
            y_raw = pos_main.tolist() + signed_neg_main.tolist() + [neutral_intensity]
            y_raw_upper = (
                pos_upper.tolist()
                + signed_neg_upper.tolist()
                + [neutral_intensity + neutral_std]
            )
            y_raw_lower = (
                pos_lower.tolist()
                + signed_neg_lower.tolist()
                + [neutral_intensity - neutral_std]
            )
            indices = list(range(len(x_raw)))
            indices.sort(key=lambda x: x_raw[x])

            x = [x_raw[i] for i in indices]
            y = [y_raw[i] for i in indices]
            y_upper = [y_raw_upper[i] for i in indices]
            y_lower = [y_raw_lower[i] for i in indices]

            lower, upper, main = draw_ci_scatter(
                x=x,
                y=y,
                y_upper=y_upper,
                y_lower=y_lower,
                name="中性刺激选择强度",
            )

            fig.add_trace(
                lower,
                row=2,
                col=3,
            )

            fig.add_trace(
                upper,
                row=2,
                col=3,
            )

            fig.add_trace(
                main,
                row=2,
                col=3,
            )

            # 添加对角线（理想一致性线）
            max_val = 9
            fig.add_trace(
                go.Scatter(
                    x=[-max_val, max_val],
                    y=[-max_val, max_val],
                    mode="lines",
                    line=dict(dash="dash", color="gray", width=2),
                    name="理想一致性线",
                    showlegend=True,
                ),
                row=2,
                col=3,
            )

            fig.update_yaxes(
                title_text="选择强度", range=[-max_val - 1, max_val + 1], row=2, col=3
            )
            fig.update_xaxes(
                title_text="标签强度（积极为正，消极为负，中性为0）", row=2, col=3
            )

    # 图2: 各标签强度对应反应时（0-9，积极和消极分开）
    if "rt_by_label_emotion" in metrics:
        rt_data = metrics["rt_by_label_emotion"].to_pandas()

        # 分离积极和消极数据
        pos_data = rt_data[rt_data["stim_type"] == "positive"]
        neg_data = rt_data[rt_data["stim_type"] == "negative"]

        # 为每个强度级别创建条形
        all_intensities = sorted(set(rt_data["label_intensity_signed"].tolist()))

        for i, intensity in enumerate(all_intensities):
            # 积极数据
            pos_intensity_data = pos_data[
                pos_data["label_intensity_signed"] == intensity
            ]
            if not pos_intensity_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=pos_intensity_data["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=pos_intensity_data["std_rt"].values,
                            visible=True,
                        ),
                        name="积极",
                        marker_color="#00cc96",
                        legendgroup="positive",
                        showlegend=False,
                        width=0.35,
                    ),
                    row=1,
                    col=2,
                )

            # 消极数据
            neg_intensity_data = neg_data[
                neg_data["label_intensity_signed"] == intensity
            ]
            if not neg_intensity_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=neg_intensity_data["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=neg_intensity_data["std_rt"].values,
                            visible=True,
                        ),
                        name="消极",
                        marker_color="#ef553b",
                        legendgroup="negative",
                        showlegend=False,
                        width=0.35,
                    ),
                    row=1,
                    col=2,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(all_intensities))),
            ticktext=[str(i) for i in all_intensities],
            row=1,
            col=2,
        )

        fig.update_yaxes(title_text="反应时(秒)", row=1, col=2)
        fig.update_xaxes(title_text="标签强度", row=1, col=2)

    # 图3: 模糊等级对应选择强度（3个等级，积极和消极分开）
    if "intensity_by_level_emotion" in metrics:
        intensity_data = metrics["intensity_by_level_emotion"].to_pandas()

        # 分离积极和消极数据
        pos_data = intensity_data[intensity_data["stim_type"] == "positive"]
        neg_data = intensity_data[intensity_data["stim_type"] == "negative"]

        # 为每个等级创建条形
        for i, level in enumerate(level_order):
            # 积极数据
            pos_level_data = pos_data[pos_data["intensity_level"] == level]
            if not pos_level_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=pos_level_data["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=pos_level_data["std_intensity"].values,
                            visible=True,
                        ),
                        name="积极",
                        marker_color="#00cc96",
                        legendgroup="positive",
                        showlegend=False,
                        width=0.35,
                    ),
                    row=1,
                    col=3,
                )

            # 消极数据
            neg_level_data = neg_data[neg_data["intensity_level"] == level]
            if not neg_level_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=neg_level_data["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=neg_level_data["std_intensity"].values,
                            visible=True,
                        ),
                        name="消极",
                        marker_color="#ef553b",
                        legendgroup="negative",
                        showlegend=False,
                        width=0.35,
                    ),
                    row=1,
                    col=3,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(3)),
            ticktext=["模糊", "中等", "清晰"],
            row=1,
            col=3,
        )

        fig.update_yaxes(title_text="选择强度", range=[0, 10], row=1, col=3)
        fig.update_xaxes(title_text="模糊等级", row=1, col=3)

    # 图4: 模糊等级对应反应时（3个等级，积极和消极分开）
    if "rt_by_level_emotion" in metrics:
        rt_data = metrics["rt_by_level_emotion"].to_pandas()

        # 分离积极和消极数据
        pos_data = rt_data[rt_data["stim_type"] == "positive"]
        neg_data = rt_data[rt_data["stim_type"] == "negative"]

        # 为每个等级创建条形
        for i, level in enumerate(level_order):
            # 积极数据
            pos_level_data = pos_data[pos_data["intensity_level"] == level]
            if not pos_level_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=pos_level_data["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=pos_level_data["std_rt"].values,
                            visible=True,
                        ),
                        name="积极",
                        marker_color="#00cc96",
                        legendgroup="positive",
                        showlegend=False,
                        width=0.35,
                    ),
                    row=2,
                    col=1,
                )

            # 消极数据
            neg_level_data = neg_data[neg_data["intensity_level"] == level]
            if not neg_level_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=neg_level_data["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=neg_level_data["std_rt"].values,
                            visible=True,
                        ),
                        name="消极",
                        marker_color="#ef553b",
                        legendgroup="negative",
                        showlegend=False,
                        width=0.35,
                    ),
                    row=2,
                    col=1,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(3)),
            ticktext=["模糊", "中等", "清晰"],
            row=2,
            col=1,
        )

        fig.update_yaxes(title_text="反应时(秒)", row=2, col=1)
        fig.update_xaxes(title_text="模糊等级", row=2, col=1)

    # 图5: 中性判定强度（散点图，积极、消极和中性分开）
    if "neutral_by_label_emotion" in metrics:
        neutral_data = metrics["neutral_by_label_emotion"].to_pandas()

        # 分离积极、消极和中性数据
        pos_data = neutral_data[neutral_data["stim_type"] == "positive"]
        neg_data = neutral_data[neutral_data["stim_type"] == "negative"]

        signed_neg_data = neg_data.copy()
        signed_neg_data["label_intensity_signed"] = -signed_neg_data[
            "label_intensity_signed"
        ]
        # signed_neg_data["neutral_proportion"] = -signed_neg_data["neutral_proportion"]

        # 添加中性刺激数据（如果存在）
        # 中性刺激的label_intensity_signed为0
        neutral_stim_data = trials_df.filter(pl.col("stim_type") == "neutral")
        if neutral_stim_data.height > 0:
            # 计算中性刺激被判断为中性的比例
            neutral_choice_count = neutral_stim_data.filter(
                pl.col("choice_type") == "neutral"
            ).height
            total_neutral_count = neutral_stim_data.height
            if total_neutral_count > 0:
                neutral_proportion = neutral_choice_count / total_neutral_count
                # 在中性位置（x=0）添加点
                # fig.add_trace(
                #     go.Scatter(
                #         x=[0],
                #         y=[neutral_proportion],
                #         mode="markers",
                #         name="中性刺激",
                #         marker=dict(size=15, color="#9467bd", symbol="diamond"),
                #         legendgroup="neutral",
                #         showlegend=True,
                #     ),
                #     row=2,
                #     col=2,
                # )

        # 积极数据（x轴为正数）
        if (
            not pos_data.empty
            and not signed_neg_data.empty
            and "neutral_proportion" in locals()
        ):
            pos_data = pos_data.sort_values("label_intensity_signed")
            signed_neg_data = signed_neg_data.sort_values("label_intensity_signed")

            x_raw = (
                pos_data["label_intensity_signed"].tolist()
                + signed_neg_data["label_intensity_signed"].tolist()
                + [0]
            )
            y_raw = (
                pos_data["neutral_proportion"].tolist()
                + signed_neg_data["neutral_proportion"].tolist()
                + [neutral_proportion]
            )

            indices = list(range(len(x_raw)))
            indices.sort(key=lambda x: x_raw[x])
            x = [x_raw[i] for i in indices]
            y = [y_raw[i] for i in indices]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name="积极刺激",
                    line=dict(width=3, color="#00cc96"),
                    marker=dict(size=10),
                    legendgroup="positive",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

    # # 消极数据（x轴为负数）
    # if not neg_data.empty:
    #     fig.add_trace(
    #         go.Scatter(
    #             x=-neg_data["label_intensity_signed"],  # x轴为负数
    #             y=neg_data["neutral_proportion"],
    #             mode="lines+markers",
    #             name="消极刺激",
    #             line=dict(width=3, color="#ef553b"),
    #             marker=dict(size=10),
    #             legendgroup="negative",
    #             showlegend=False,
    #         ),
    #         row=2,
    #         col=2,
    #     )

    fig.update_yaxes(title_text="中性判定比例", range=[0, 1.2], row=2, col=2)
    fig.update_xaxes(title_text="标签强度（积极为正，消极为负，中性为0）", row=2, col=2)

    # 调整布局
    fig.update_layout(
        height=1400,
        width=1800,
        title=dict(
            text="面部情绪识别实验单人分析报告",
            font=dict(size=24, family="Arial Black"),
            x=0.5,
        ),
        showlegend=True,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        barmode="group",
    )

    html_path = result_dir / "emotion_face_single_visualization.html"
    fig.write_html(str(html_path))
    figs.append(fig)

    return figs


def save_results(
    metrics: dict[str, Any],
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

    # 保存按标签和情绪分组的强度数据
    if "intensity_by_label_emotion" in metrics:
        intensity_by_label_emotion_df = metrics[
            "intensity_by_label_emotion"
        ].to_pandas()
        intensity_by_label_emotion_path = (
            result_dir / "emotion_face_intensity_by_label_emotion.csv"
        )
        intensity_by_label_emotion_df.to_csv(
            intensity_by_label_emotion_path, index=False
        )
        saved_files["intensity_by_label_emotion"] = intensity_by_label_emotion_path

    # 保存按标签和情绪分组的反应时数据
    if "rt_by_label_emotion" in metrics:
        rt_by_label_emotion_df = metrics["rt_by_label_emotion"].to_pandas()
        rt_by_label_emotion_path = result_dir / "emotion_face_rt_by_label_emotion.csv"
        rt_by_label_emotion_df.to_csv(rt_by_label_emotion_path, index=False)
        saved_files["rt_by_label_emotion"] = rt_by_label_emotion_path

    # 保存按等级和情绪分组的强度数据
    if "intensity_by_level_emotion" in metrics:
        intensity_by_level_emotion_df = metrics[
            "intensity_by_level_emotion"
        ].to_pandas()
        intensity_by_level_emotion_path = (
            result_dir / "emotion_face_intensity_by_level_emotion.csv"
        )
        intensity_by_level_emotion_df.to_csv(
            intensity_by_level_emotion_path, index=False
        )
        saved_files["intensity_by_level_emotion"] = intensity_by_level_emotion_path

    # 保存按等级和情绪分组的反应时数据
    if "rt_by_level_emotion" in metrics:
        rt_by_level_emotion_df = metrics["rt_by_level_emotion"].to_pandas()
        rt_by_level_emotion_path = result_dir / "emotion_face_rt_by_level_emotion.csv"
        rt_by_level_emotion_df.to_csv(rt_by_level_emotion_path, index=False)
        saved_files["rt_by_level_emotion"] = rt_by_level_emotion_path

    # 保存中性判定数据
    if "neutral_by_label_emotion" in metrics:
        neutral_by_label_emotion_df = metrics["neutral_by_label_emotion"].to_pandas()
        neutral_by_label_emotion_path = (
            result_dir / "emotion_face_neutral_by_label_emotion.csv"
        )
        neutral_by_label_emotion_df.to_csv(neutral_by_label_emotion_path, index=False)
        saved_files["neutral_by_label_emotion"] = neutral_by_label_emotion_path

    return saved_files


def analyze_emotion_face_data(
    df: pl.DataFrame,
    target_blocks: list[int] = [0, 1],
    result_dir: Path = Path("results"),
) -> dict[str, Any]:
    """分析单个被试的面部情绪识别数据"""
    # 1. 加载和预处理数据
    trials_df = load_and_preprocess_data(df)

    if trials_df is None:
        return None

    # 2. 计算基本指标
    metrics = calculate_basic_metrics(trials_df)

    # 3. 计算关键指标
    key_metrics_dict = calculate_key_metrics(trials_df, metrics)

    # 4. 计算全面指标
    comprehensive_metrics = calculate_comprehensive_key_metrics(trials_df, metrics)

    # 5. 创建单人可视化
    _ = create_single_visualizations(trials_df, metrics, key_metrics_dict, result_dir)

    # 6. 保存结果
    saved_files = save_results(metrics, key_metrics_dict, result_dir)

    # 7. 生成报告
    report = {
        "data_summary": {
            "total_trials": trials_df.height,
            "num_blocks": len(trials_df["block_index"].unique()),
            "overall_accuracy": float(metrics["overall_accuracy"]),
            "median_rt": float(metrics["median_rt"]),
        },
        "key_metrics": key_metrics_dict,
        "comprehensive_metrics": comprehensive_metrics,
        "metrics": metrics,
        "saved_files": saved_files,
        "trials_df": trials_df,
    }

    print(f"\n✅ 单人分析完成！结果保存在: {result_dir}")
    return report


def create_single_group_visualizations(
    group_metrics: list[dict[str, float]],
    group_trials: list[pl.DataFrame] = None,
    group_name: str = None,
    result_dir: Path = None,
) -> list[go.Figure]:
    """单个组的组分析可视化 - 组平均版本，结构与单人分析相同"""
    figs = []

    # 如果没有试次数据，直接返回
    if not group_trials or len(group_trials) == 0:
        return figs

    new_group_trials = []
    for trials_df in group_trials:
        new_group_trials.append(trials_df.drop("intensity_rt", strict=False))

    group_trials, new_group_trials = new_group_trials, group_trials

    # 1. 合并所有被试的数据，计算组平均
    all_trials = pl.concat(group_trials)

    # 2. 计算组级别的指标（与单人分析相同）
    group_full_metrics = calculate_basic_metrics(all_trials)

    if not group_full_metrics.get("has_correct_data", False):
        print("⚠️ 组数据中没有正确的试次，无法生成可视化")
        return figs

    # 3. 计算关键指标
    group_key_metrics = calculate_key_metrics(all_trials, group_full_metrics)

    # 4. 创建组平均可视化 - 使用与单人分析完全相同的布局
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "各标签强度对应选择强度（组平均）",
            "各标签强度对应反应时（组平均）",
            "模糊等级对应选择强度（组平均）",
            "模糊等级对应反应时（组平均）",
            "中性判定强度（组平均）",
            "强度一致性分析（组平均）",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # 定义等级顺序
    level_order = ["blur", "mid", "clear"]

    # 图1: 各标签强度对应选择强度（0-9，积极和消极分开） - 组平均
    if "intensity_by_label_emotion" in group_full_metrics:
        intensity_data = group_full_metrics["intensity_by_label_emotion"].to_pandas()

        # 分离积极和消极数据
        pos_data = intensity_data[intensity_data["stim_type"] == "positive"]
        neg_data = intensity_data[intensity_data["stim_type"] == "negative"]

        # 为每个强度级别创建条形
        all_intensities = sorted(set(intensity_data["label_intensity_signed"].tolist()))

        # 存储x轴位置
        for i, intensity in enumerate(all_intensities):
            # 积极数据
            pos_intensity_data = pos_data[
                pos_data["label_intensity_signed"] == intensity
            ]
            if not pos_intensity_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=pos_intensity_data["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=pos_intensity_data["std_intensity"].values,
                            visible=True,
                        ),
                        name="积极" if intensity == all_intensities[0] else "",
                        marker_color="#00cc96",
                        legendgroup="positive",
                        showlegend=intensity == all_intensities[0],
                        width=0.35,
                    ),
                    row=1,
                    col=1,
                )

            # 消极数据
            neg_intensity_data = neg_data[
                neg_data["label_intensity_signed"] == intensity
            ]
            if not neg_intensity_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=neg_intensity_data["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=neg_intensity_data["std_intensity"].values,
                            visible=True,
                        ),
                        name="消极" if intensity == all_intensities[0] else "",
                        marker_color="#ef553b",
                        legendgroup="negative",
                        showlegend=intensity == all_intensities[0],
                        width=0.35,
                    ),
                    row=1,
                    col=1,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(all_intensities))),
            ticktext=[str(i) for i in all_intensities],
            row=1,
            col=1,
        )

        fig.update_yaxes(title_text="选择强度", range=[0, 10], row=1, col=1)
        fig.update_xaxes(title_text="标签强度", row=1, col=1)
        # 图6: 强度一致性分析（积极、消极和中性放在一张图中）
        # 处理中性刺激数据
        neutral_stim_data = all_trials.filter(pl.col("stim_type") == "neutral")
        if neutral_stim_data.height > 0:
            # 计算中性刺激的平均选择强度（使用正确试次）
            correct_neutral_trials = neutral_stim_data.filter(pl.col("correct"))
            if correct_neutral_trials.height > 0:
                neutral_intensity = correct_neutral_trials["intensity"].mean()
                neutral_std = correct_neutral_trials["intensity"].std()
                neutral_n = correct_neutral_trials.height

        # 中性刺激数据（如果存在）
        if (
            not pos_data.empty
            and not neg_data.empty
            and "neutral_intensity" in locals()
            and neutral_n > 0
        ):
            pos_data = pos_data.sort_values("label_intensity_signed")
            pos_upper = pos_data["mean_intensity"] + pos_data["std_intensity"]
            pos_lower = pos_data["mean_intensity"] - pos_data["std_intensity"]
            pos_main = pos_data["mean_intensity"]

            # 处理消极数据（将x轴变为负数，y轴保持不变）
            neg_data_sorted = neg_data.sort_values("label_intensity_signed")
            neg_data_sorted = neg_data_sorted.copy()
            neg_data_sorted["label_intensity_signed"] = -neg_data_sorted[
                "label_intensity_signed"
            ]
            neg_data_sorted["mean_intensity"] = -neg_data_sorted["mean_intensity"]
            neg_data_sorted["std_intensity"] = -neg_data_sorted["std_intensity"]

            neg_upper = (
                neg_data_sorted["mean_intensity"] + neg_data_sorted["std_intensity"]
            )
            neg_lower = (
                neg_data_sorted["mean_intensity"] - neg_data_sorted["std_intensity"]
            )
            neg_main = neg_data_sorted["mean_intensity"]

            # 合并所有数据点
            x_raw = (
                pos_data["label_intensity_signed"].tolist()
                + neg_data_sorted["label_intensity_signed"].tolist()
                + [0]
            )
            y_raw = pos_main.tolist() + neg_main.tolist() + [neutral_intensity]
            y_raw_upper = (
                pos_upper.tolist()
                + neg_upper.tolist()
                + [neutral_intensity + neutral_std]
            )
            y_raw_lower = (
                pos_lower.tolist()
                + neg_lower.tolist()
                + [neutral_intensity - neutral_std]
            )

            # 按x值排序
            indices = list(range(len(x_raw)))
            indices.sort(key=lambda x: x_raw[x])

            x = [x_raw[i] for i in indices]
            y = [y_raw[i] for i in indices]
            y_upper = [y_raw_upper[i] for i in indices]
            y_lower = [y_raw_lower[i] for i in indices]

            # 绘制置信区间和主线条
            lower, upper, main = draw_ci_scatter(
                x=x,
                y=y,
                y_upper=y_upper,
                y_lower=y_lower,
                name="组平均选择强度",
            )

            fig.add_trace(lower, row=2, col=3)
            fig.add_trace(upper, row=2, col=3)
            fig.add_trace(main, row=2, col=3)

            # 添加对角线（理想一致性线）
            max_val = 9
            fig.add_trace(
                go.Scatter(
                    x=[-max_val, max_val],
                    y=[-max_val, max_val],
                    mode="lines",
                    line=dict(dash="dash", color="gray", width=2),
                    name="理想一致性线",
                    showlegend=True,
                ),
                row=2,
                col=3,
            )

            fig.update_yaxes(
                title_text="选择强度", range=[-max_val - 1, max_val + 1], row=2, col=3
            )
            fig.update_xaxes(
                title_text="标签强度（积极为正，消极为负，中性为0）", row=2, col=3
            )

    # 图2: 各标签强度对应反应时（0-9，积极和消极分开） - 组平均
    if "rt_by_label_emotion" in group_full_metrics:
        rt_data = group_full_metrics["rt_by_label_emotion"].to_pandas()

        # 分离积极和消极数据
        pos_data = rt_data[rt_data["stim_type"] == "positive"]
        neg_data = rt_data[rt_data["stim_type"] == "negative"]

        # 为每个强度级别创建条形
        all_intensities = sorted(set(rt_data["label_intensity_signed"].tolist()))

        for i, intensity in enumerate(all_intensities):
            # 积极数据
            pos_intensity_data = pos_data[
                pos_data["label_intensity_signed"] == intensity
            ]
            if not pos_intensity_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=pos_intensity_data["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=pos_intensity_data["std_rt"].values,
                            visible=True,
                        ),
                        name="积极",
                        marker_color="#00cc96",
                        legendgroup="positive",
                        showlegend=False,
                        width=0.35,
                    ),
                    row=1,
                    col=2,
                )

            # 消极数据
            neg_intensity_data = neg_data[
                neg_data["label_intensity_signed"] == intensity
            ]
            if not neg_intensity_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=neg_intensity_data["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=neg_intensity_data["std_rt"].values,
                            visible=True,
                        ),
                        name="消极",
                        marker_color="#ef553b",
                        legendgroup="negative",
                        showlegend=False,
                        width=0.35,
                    ),
                    row=1,
                    col=2,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(all_intensities))),
            ticktext=[str(i) for i in all_intensities],
            row=1,
            col=2,
        )

        fig.update_yaxes(title_text="反应时(秒)", row=1, col=2)
        fig.update_xaxes(title_text="标签强度", row=1, col=2)

    # 图3: 模糊等级对应选择强度（3个等级，积极和消极分开） - 组平均
    if "intensity_by_level_emotion" in group_full_metrics:
        intensity_data = group_full_metrics["intensity_by_level_emotion"].to_pandas()

        # 分离积极和消极数据
        pos_data = intensity_data[intensity_data["stim_type"] == "positive"]
        neg_data = intensity_data[intensity_data["stim_type"] == "negative"]

        # 为每个等级创建条形
        for i, level in enumerate(level_order):
            # 积极数据
            pos_level_data = pos_data[pos_data["intensity_level"] == level]
            if not pos_level_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=pos_level_data["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=pos_level_data["std_intensity"].values,
                            visible=True,
                        ),
                        name="积极",
                        marker_color="#00cc96",
                        legendgroup="positive",
                        showlegend=False,
                        width=0.35,
                    ),
                    row=1,
                    col=3,
                )

            # 消极数据
            neg_level_data = neg_data[neg_data["intensity_level"] == level]
            if not neg_level_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=neg_level_data["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=neg_level_data["std_intensity"].values,
                            visible=True,
                        ),
                        name="消极",
                        marker_color="#ef553b",
                        legendgroup="negative",
                        showlegend=False,
                        width=0.35,
                    ),
                    row=1,
                    col=3,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(3)),
            ticktext=["模糊", "中等", "清晰"],
            row=1,
            col=3,
        )

        fig.update_yaxes(title_text="选择强度", range=[0, 10], row=1, col=3)
        fig.update_xaxes(title_text="模糊等级", row=1, col=3)

    # 图4: 模糊等级对应反应时（3个等级，积极和消极分开） - 组平均
    if "rt_by_level_emotion" in group_full_metrics:
        rt_data = group_full_metrics["rt_by_level_emotion"].to_pandas()

        # 分离积极和消极数据
        pos_data = rt_data[rt_data["stim_type"] == "positive"]
        neg_data = rt_data[rt_data["stim_type"] == "negative"]

        # 为每个等级创建条形
        for i, level in enumerate(level_order):
            # 积极数据
            pos_level_data = pos_data[pos_data["intensity_level"] == level]
            if not pos_level_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=pos_level_data["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=pos_level_data["std_rt"].values,
                            visible=True,
                        ),
                        name="积极",
                        marker_color="#00cc96",
                        legendgroup="positive",
                        showlegend=False,
                        width=0.35,
                    ),
                    row=2,
                    col=1,
                )

            # 消极数据
            neg_level_data = neg_data[neg_data["intensity_level"] == level]
            if not neg_level_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=neg_level_data["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=neg_level_data["std_rt"].values,
                            visible=True,
                        ),
                        name="消极",
                        marker_color="#ef553b",
                        legendgroup="negative",
                        showlegend=False,
                        width=0.35,
                    ),
                    row=2,
                    col=1,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(3)),
            ticktext=["模糊", "中等", "清晰"],
            row=2,
            col=1,
        )

        fig.update_yaxes(title_text="反应时(秒)", row=2, col=1)
        fig.update_xaxes(title_text="模糊等级", row=2, col=1)

    # 图5: 中性判定强度（积极、消极和中性分开） - 组平均
    if "neutral_by_label_emotion" in group_full_metrics:
        neutral_data = group_full_metrics["neutral_by_label_emotion"].to_pandas()

        # 分离积极和消极数据
        pos_data = neutral_data[neutral_data["stim_type"] == "positive"]
        neg_data = neutral_data[neutral_data["stim_type"] == "negative"]

        # 处理消极数据（将x轴变为负数）
        neg_data_sorted = neg_data.copy()
        neg_data_sorted["label_intensity_signed"] = -neg_data_sorted[
            "label_intensity_signed"
        ]
        neg_data_sorted = neg_data_sorted.sort_values("label_intensity_signed")

        # 处理积极数据
        pos_data_sorted = pos_data.sort_values("label_intensity_signed")

        # 添加中性刺激数据
        neutral_stim_data = all_trials.filter(pl.col("stim_type") == "neutral")
        if neutral_stim_data.height > 0:
            # 计算中性刺激被判断为中性的比例
            neutral_choice_count = neutral_stim_data.filter(
                pl.col("choice_type") == "neutral"
            ).height
            total_neutral_count = neutral_stim_data.height
            if total_neutral_count > 0:
                neutral_proportion = neutral_choice_count / total_neutral_count

        # 合并所有数据并绘制
        if (
            not pos_data_sorted.empty
            and not neg_data_sorted.empty
            and "neutral_proportion" in locals()
        ):
            # 合并所有x和y数据
            x_all = (
                pos_data_sorted["label_intensity_signed"].tolist()
                + neg_data_sorted["label_intensity_signed"].tolist()
                + [0]
            )
            y_all = (
                pos_data_sorted["neutral_proportion"].tolist()
                + neg_data_sorted["neutral_proportion"].tolist()
                + [neutral_proportion]
            )

            # 按x值排序
            combined_data = list(zip(x_all, y_all))
            combined_data.sort(key=lambda x: x[0])
            x_sorted = [item[0] for item in combined_data]
            y_sorted = [item[1] for item in combined_data]

            # 绘制线条
            fig.add_trace(
                go.Scatter(
                    x=x_sorted,
                    y=y_sorted,
                    mode="lines+markers",
                    line=dict(width=3, color="#9467bd"),
                    marker=dict(size=10),
                    name="中性判定强度",
                    showlegend=True,
                ),
                row=2,
                col=2,
            )

            # 标记不同刺激类型
            # 积极刺激点
            if not pos_data_sorted.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pos_data_sorted["label_intensity_signed"],
                        y=pos_data_sorted["neutral_proportion"],
                        mode="markers",
                        name="积极刺激",
                        marker=dict(size=10, color="#00cc96", symbol="circle"),
                        showlegend=True,
                    ),
                    row=2,
                    col=2,
                )

            # 消极刺激点
            if not neg_data_sorted.empty:
                fig.add_trace(
                    go.Scatter(
                        x=neg_data_sorted["label_intensity_signed"],
                        y=neg_data_sorted["neutral_proportion"],
                        mode="markers",
                        name="消极刺激",
                        marker=dict(size=10, color="#ef553b", symbol="circle"),
                        showlegend=True,
                    ),
                    row=2,
                    col=2,
                )

            # 中性刺激点
            if "neutral_proportion" in locals():
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[neutral_proportion],
                        mode="markers",
                        name="中性刺激",
                        marker=dict(size=15, color="#9467bd", symbol="diamond"),
                        showlegend=True,
                    ),
                    row=2,
                    col=2,
                )

        fig.update_yaxes(title_text="中性判定比例", range=[0, 1.2], row=2, col=2)
        fig.update_xaxes(
            title_text="标签强度（积极为正，消极为负，中性为0）", row=2, col=2
        )

    # 调整布局
    fig.update_layout(
        height=1400,
        width=1800,
        title=dict(
            text=f"面部情绪识别{group_name}组分析报告（组平均）"
            if group_name
            else "面部情绪识别组分析报告（组平均）",
            font=dict(size=24, family="Arial Black"),
            x=0.5,
        ),
        showlegend=True,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        barmode="group",
    )

    figs.append(fig)

    return figs


def create_multi_group_visualizations(
    control_metrics: list[dict[str, float]],
    experimental_metrics: list[dict[str, float]],
    control_trials: list[pl.DataFrame] = None,
    experimental_trials: list[pl.DataFrame] = None,
    control_name: str = "对照组",
    experimental_name: str = "实验组",
) -> list[go.Figure]:
    """多组比较可视化 - 两组对比版本，结构与单人分析相同"""
    figs = []

    # 如果没有试次数据，直接返回
    if not control_trials or not experimental_trials:
        return figs
    new_control_trials = []
    for trials_df in control_trials:
        new_control_trials.append(trials_df.drop("intensity_rt", strict=False))
    control_trials, new_control_trials = new_control_trials, control_trials

    new_experimental_trials = []
    for trials_df in experimental_trials:
        new_experimental_trials.append(trials_df.drop("intensity_rt", strict=False))
    experimental_trials, new_experimental_trials = (
        new_experimental_trials,
        experimental_trials,
    )

    # 1. 合并每组的数据，分别计算组平均
    control_all_trials = pl.concat(control_trials)
    experimental_all_trials = pl.concat(experimental_trials)

    # 2. 分别计算两组的指标
    control_full_metrics = calculate_basic_metrics(control_all_trials)
    experimental_full_metrics = calculate_basic_metrics(experimental_all_trials)

    if not control_full_metrics.get(
        "has_correct_data", False
    ) or not experimental_full_metrics.get("has_correct_data", False):
        print("⚠️ 至少一组数据中没有正确的试次，无法生成多组对比可视化")
        return figs

    # 3. 创建多组对比可视化 - 使用4行3列布局，将积极和消极拆分为不同子图
    fig = make_subplots(
        rows=4,  # 增加行数
        cols=3,
        subplot_titles=(
            # 第一行：积极情绪相关
            "各标签强度对应选择强度（积极）",
            "各标签强度对应反应时（积极）",
            "模糊等级对应选择强度（积极）",
            # 第二行：消极情绪相关
            "各标签强度对应选择强度（消极）",
            "各标签强度对应反应时（消极）",
            "模糊等级对应选择强度（消极）",
            # 第三行：反应时和中性判定
            "模糊等级对应反应时（积极）",
            "模糊等级对应反应时（消极）",
            # 第四行：强度一致性分析
            "中性判定强度（组间对比）",
            "强度一致性分析",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],  # 第一行
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],  # 第二行
            [{"type": "bar"}, {"type": "bar"}, None],  # 第三行
            [{"type": "scatter"}, {"type": "scatter"}, None],  # 第四行
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    fig = make_subplots(
        rows=6,  # 增加行数以容纳总体图
        cols=3,
        subplot_titles=(
            # 第一行：积极情绪相关
            "各标签强度对应选择强度（积极）",
            "各标签强度对应反应时（积极）",
            "模糊等级对应选择强度（积极）",
            # 第二行：消极情绪相关
            "各标签强度对应选择强度（消极）",
            "各标签强度对应反应时（消极）",
            "模糊等级对应选择强度（消极）",
            # 第三行：反应时和中性判定
            "模糊等级对应反应时（积极）",
            "模糊等级对应反应时（消极）",
            # 第四行：强度一致性分析
            "中性判定强度（组间对比）",
            "强度一致性分析",
            # 第五行：新增总体图3-4
            "各标签强度对应选择强度（总体）",  # 新增总体图1
            "各标签强度对应反应时（总体）",  # 新增总体图2
            # 第六行：新增总体图5-6
            "模糊等级对应选择强度（总体）",
            "模糊等级对应反应时（总体）",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],  # 第一行
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],  # 第二行
            [{"type": "bar"}, {"type": "bar"}, None],  # 第三行
            [{"type": "scatter"}, {"type": "scatter"}, None],  # 第四行
            [{"type": "bar"}, {"type": "bar"}, None],  # 第五行
            [{"type": "bar"}, {"type": "bar"}, None],  # 第六行
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
    )
    # 颜色定义
    control_color = "#1f77b4"  # 蓝色
    experimental_color = "#ff7f0e"  # 橙色

    # 定义图案区分积极和消极
    control_negative_pattern = dict(shape="/", size=5, solidity=0.7)
    experimental_negative_pattern = dict(shape="\\", size=5, solidity=0.7)

    # 定义等级顺序
    level_order = ["blur", "mid", "clear"]

    # 图1: 各标签强度对应选择强度（积极）- 两组对比
    if (
        "intensity_by_label_emotion" in control_full_metrics
        and "intensity_by_label_emotion" in experimental_full_metrics
    ):
        control_intensity_data = control_full_metrics[
            "intensity_by_label_emotion"
        ].to_pandas()
        experimental_intensity_data = experimental_full_metrics[
            "intensity_by_label_emotion"
        ].to_pandas()

        # 只提取积极数据
        control_pos_data = control_intensity_data[
            control_intensity_data["stim_type"] == "positive"
        ]
        experimental_pos_data = experimental_intensity_data[
            experimental_intensity_data["stim_type"] == "positive"
        ]

        # 获取所有强度级别
        all_intensities = sorted(
            set(
                control_intensity_data["label_intensity_signed"].tolist()
                + experimental_intensity_data["label_intensity_signed"].tolist()
            )
        )

        # 只保留积极强度（0-9）
        all_intensities = [i for i in all_intensities if i >= 0]

        # 为每个强度级别创建条形
        for i, intensity in enumerate(all_intensities):
            # 对照组积极数据
            control_pos_intensity = control_pos_data[
                control_pos_data["label_intensity_signed"] == intensity
            ]
            if not control_pos_intensity.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=control_pos_intensity["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=control_pos_intensity["std_intensity"].values,
                            visible=True,
                        ),
                        name=f"{control_name}积极"
                        if intensity == all_intensities[0]
                        else "",
                        marker_color=control_color,
                        marker=dict(opacity=0.8),
                        legendgroup=f"{control_name}-positive",
                        showlegend=intensity == all_intensities[0],
                        width=0.4,
                    ),
                    row=1,
                    col=1,
                )

            # 实验组积极数据
            experimental_pos_intensity = experimental_pos_data[
                experimental_pos_data["label_intensity_signed"] == intensity
            ]
            if not experimental_pos_intensity.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=experimental_pos_intensity["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=experimental_pos_intensity["std_intensity"].values,
                            visible=True,
                        ),
                        name=f"{experimental_name}积极"
                        if intensity == all_intensities[0]
                        else "",
                        marker_color=experimental_color,
                        marker=dict(opacity=0.8),
                        legendgroup=f"{experimental_name}-positive",
                        showlegend=intensity == all_intensities[0],
                        width=0.4,
                    ),
                    row=1,
                    col=1,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(all_intensities))),
            ticktext=[str(i) for i in all_intensities],
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="选择强度", range=[0, 10], row=1, col=1)
        fig.update_xaxes(title_text="标签强度", row=1, col=1)

    # 图2: 各标签强度对应反应时（积极）- 两组对比
    if (
        "rt_by_label_emotion" in control_full_metrics
        and "rt_by_label_emotion" in experimental_full_metrics
    ):
        control_rt_data = control_full_metrics["rt_by_label_emotion"].to_pandas()
        experimental_rt_data = experimental_full_metrics[
            "rt_by_label_emotion"
        ].to_pandas()

        # 只提取积极数据
        control_pos_data = control_rt_data[control_rt_data["stim_type"] == "positive"]
        experimental_pos_data = experimental_rt_data[
            experimental_rt_data["stim_type"] == "positive"
        ]

        # 获取所有强度级别
        all_intensities = sorted(
            set(
                control_rt_data["label_intensity_signed"].tolist()
                + experimental_rt_data["label_intensity_signed"].tolist()
            )
        )

        # 只保留积极强度（0-9）
        all_intensities = [i for i in all_intensities if i >= 0]

        for i, intensity in enumerate(all_intensities):
            # 对照组积极数据
            control_pos_intensity = control_pos_data[
                control_pos_data["label_intensity_signed"] == intensity
            ]
            if not control_pos_intensity.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=control_pos_intensity["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=control_pos_intensity["std_rt"].values,
                            visible=True,
                        ),
                        name=f"{control_name}积极",
                        marker_color=control_color,
                        marker=dict(opacity=0.8),
                        legendgroup=f"{control_name}-positive",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=1,
                    col=2,
                )

            # 实验组积极数据
            experimental_pos_intensity = experimental_pos_data[
                experimental_pos_data["label_intensity_signed"] == intensity
            ]
            if not experimental_pos_intensity.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=experimental_pos_intensity["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=experimental_pos_intensity["std_rt"].values,
                            visible=True,
                        ),
                        name=f"{experimental_name}积极",
                        marker_color=experimental_color,
                        marker=dict(opacity=0.8),
                        legendgroup=f"{experimental_name}-positive",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=1,
                    col=2,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(all_intensities))),
            ticktext=[str(i) for i in all_intensities],
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="反应时(秒)", row=1, col=2)
        fig.update_xaxes(title_text="标签强度", row=1, col=2)

    # 图3: 模糊等级对应选择强度（积极）- 两组对比
    if (
        "intensity_by_level_emotion" in control_full_metrics
        and "intensity_by_level_emotion" in experimental_full_metrics
    ):
        control_intensity_data = control_full_metrics[
            "intensity_by_level_emotion"
        ].to_pandas()
        experimental_intensity_data = experimental_full_metrics[
            "intensity_by_level_emotion"
        ].to_pandas()

        # 只提取积极数据
        control_pos_data = control_intensity_data[
            control_intensity_data["stim_type"] == "positive"
        ]
        experimental_pos_data = experimental_intensity_data[
            experimental_intensity_data["stim_type"] == "positive"
        ]

        # 为每个等级创建条形
        for i, level in enumerate(level_order):
            # 对照组积极数据
            control_pos_level = control_pos_data[
                control_pos_data["intensity_level"] == level
            ]
            if not control_pos_level.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=control_pos_level["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=control_pos_level["std_intensity"].values,
                            visible=True,
                        ),
                        name=f"{control_name}积极",
                        marker_color=control_color,
                        marker=dict(opacity=0.8),
                        legendgroup=f"{control_name}-positive",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=1,
                    col=3,
                )

            # 实验组积极数据
            experimental_pos_level = experimental_pos_data[
                experimental_pos_data["intensity_level"] == level
            ]
            if not experimental_pos_level.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=experimental_pos_level["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=experimental_pos_level["std_intensity"].values,
                            visible=True,
                        ),
                        name=f"{experimental_name}积极",
                        marker_color=experimental_color,
                        marker=dict(opacity=0.8),
                        legendgroup=f"{experimental_name}-positive",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=1,
                    col=3,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(3)),
            ticktext=["模糊", "中等", "清晰"],
            row=1,
            col=3,
        )
        fig.update_yaxes(title_text="选择强度", range=[0, 10], row=1, col=3)
        fig.update_xaxes(title_text="模糊等级", row=1, col=3)

    # 图4: 各标签强度对应选择强度（消极）- 两组对比
    if (
        "intensity_by_label_emotion" in control_full_metrics
        and "intensity_by_label_emotion" in experimental_full_metrics
    ):
        control_intensity_data = control_full_metrics[
            "intensity_by_label_emotion"
        ].to_pandas()
        experimental_intensity_data = experimental_full_metrics[
            "intensity_by_label_emotion"
        ].to_pandas()

        # 只提取消极数据
        control_neg_data = control_intensity_data[
            control_intensity_data["stim_type"] == "negative"
        ]
        experimental_neg_data = experimental_intensity_data[
            experimental_intensity_data["stim_type"] == "negative"
        ]

        # 获取所有强度级别
        all_intensities = sorted(
            set(
                control_intensity_data["label_intensity_signed"].tolist()
                + experimental_intensity_data["label_intensity_signed"].tolist()
            )
        )

        # 只保留消极强度（0-9），但显示为正数
        all_intensities = [i for i in all_intensities if i >= 0]

        # 为每个强度级别创建条形
        for i, intensity in enumerate(all_intensities):
            # 对照组消极数据
            control_neg_intensity = control_neg_data[
                control_neg_data["label_intensity_signed"] == intensity
            ]
            if not control_neg_intensity.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=control_neg_intensity["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=control_neg_intensity["std_intensity"].values,
                            visible=True,
                        ),
                        name=f"{control_name}消极"
                        if intensity == all_intensities[0]
                        else "",
                        marker_color=control_color,
                        marker=dict(opacity=0.8, pattern=control_negative_pattern),
                        legendgroup=f"{control_name}-negative",
                        showlegend=intensity == all_intensities[0],
                        width=0.4,
                    ),
                    row=2,
                    col=1,
                )

            # 实验组消极数据
            experimental_neg_intensity = experimental_neg_data[
                experimental_neg_data["label_intensity_signed"] == intensity
            ]
            if not experimental_neg_intensity.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=experimental_neg_intensity["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=experimental_neg_intensity["std_intensity"].values,
                            visible=True,
                        ),
                        name=f"{experimental_name}消极"
                        if intensity == all_intensities[0]
                        else "",
                        marker_color=experimental_color,
                        marker=dict(opacity=0.8, pattern=experimental_negative_pattern),
                        legendgroup=f"{experimental_name}-negative",
                        showlegend=intensity == all_intensities[0],
                        width=0.4,
                    ),
                    row=2,
                    col=1,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(all_intensities))),
            ticktext=[str(i) for i in all_intensities],
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="选择强度", range=[0, 10], row=2, col=1)
        fig.update_xaxes(title_text="标签强度", row=2, col=1)

    # 图5: 各标签强度对应反应时（消极）- 两组对比
    if (
        "rt_by_label_emotion" in control_full_metrics
        and "rt_by_label_emotion" in experimental_full_metrics
    ):
        control_rt_data = control_full_metrics["rt_by_label_emotion"].to_pandas()
        experimental_rt_data = experimental_full_metrics[
            "rt_by_label_emotion"
        ].to_pandas()

        # 只提取消极数据
        control_neg_data = control_rt_data[control_rt_data["stim_type"] == "negative"]
        experimental_neg_data = experimental_rt_data[
            experimental_rt_data["stim_type"] == "negative"
        ]

        # 获取所有强度级别
        all_intensities = sorted(
            set(
                control_rt_data["label_intensity_signed"].tolist()
                + experimental_rt_data["label_intensity_signed"].tolist()
            )
        )

        # 只保留消极强度（0-9）
        all_intensities = [i for i in all_intensities if i >= 0]

        for i, intensity in enumerate(all_intensities):
            # 对照组消极数据
            control_neg_intensity = control_neg_data[
                control_neg_data["label_intensity_signed"] == intensity
            ]
            if not control_neg_intensity.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=control_neg_intensity["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=control_neg_intensity["std_rt"].values,
                            visible=True,
                        ),
                        name=f"{control_name}消极",
                        marker_color=control_color,
                        marker=dict(opacity=0.8, pattern=control_negative_pattern),
                        legendgroup=f"{control_name}-negative",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=2,
                    col=2,
                )

            # 实验组消极数据
            experimental_neg_intensity = experimental_neg_data[
                experimental_neg_data["label_intensity_signed"] == intensity
            ]
            if not experimental_neg_intensity.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=experimental_neg_intensity["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=experimental_neg_intensity["std_rt"].values,
                            visible=True,
                        ),
                        name=f"{experimental_name}消极",
                        marker_color=experimental_color,
                        marker=dict(opacity=0.8, pattern=experimental_negative_pattern),
                        legendgroup=f"{experimental_name}-negative",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=2,
                    col=2,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(all_intensities))),
            ticktext=[str(i) for i in all_intensities],
            row=2,
            col=2,
        )
        fig.update_yaxes(title_text="反应时(秒)", row=2, col=2)
        fig.update_xaxes(title_text="标签强度", row=2, col=2)

    # 图6: 模糊等级对应选择强度（消极）- 两组对比
    if (
        "intensity_by_level_emotion" in control_full_metrics
        and "intensity_by_level_emotion" in experimental_full_metrics
    ):
        control_intensity_data = control_full_metrics[
            "intensity_by_level_emotion"
        ].to_pandas()
        experimental_intensity_data = experimental_full_metrics[
            "intensity_by_level_emotion"
        ].to_pandas()

        # 只提取消极数据
        control_neg_data = control_intensity_data[
            control_intensity_data["stim_type"] == "negative"
        ]
        experimental_neg_data = experimental_intensity_data[
            experimental_intensity_data["stim_type"] == "negative"
        ]

        # 为每个等级创建条形
        for i, level in enumerate(level_order):
            # 对照组消极数据
            control_neg_level = control_neg_data[
                control_neg_data["intensity_level"] == level
            ]
            if not control_neg_level.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=control_neg_level["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=control_neg_level["std_intensity"].values,
                            visible=True,
                        ),
                        name=f"{control_name}消极",
                        marker_color=control_color,
                        marker=dict(opacity=0.8, pattern=control_negative_pattern),
                        legendgroup=f"{control_name}-negative",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=2,
                    col=3,
                )

            # 实验组消极数据
            experimental_neg_level = experimental_neg_data[
                experimental_neg_data["intensity_level"] == level
            ]
            if not experimental_neg_level.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=experimental_neg_level["mean_intensity"].values,
                        error_y=dict(
                            type="data",
                            array=experimental_neg_level["std_intensity"].values,
                            visible=True,
                        ),
                        name=f"{experimental_name}消极",
                        marker_color=experimental_color,
                        marker=dict(opacity=0.8, pattern=experimental_negative_pattern),
                        legendgroup=f"{experimental_name}-negative",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=2,
                    col=3,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(3)),
            ticktext=["模糊", "中等", "清晰"],
            row=2,
            col=3,
        )
        fig.update_yaxes(title_text="选择强度", range=[0, 10], row=2, col=3)
        fig.update_xaxes(title_text="模糊等级", row=2, col=3)

    # 图7: 模糊等级对应反应时（积极）- 两组对比
    if (
        "rt_by_level_emotion" in control_full_metrics
        and "rt_by_level_emotion" in experimental_full_metrics
    ):
        control_rt_data = control_full_metrics["rt_by_level_emotion"].to_pandas()
        experimental_rt_data = experimental_full_metrics[
            "rt_by_level_emotion"
        ].to_pandas()

        # 只提取积极数据
        control_pos_data = control_rt_data[control_rt_data["stim_type"] == "positive"]
        experimental_pos_data = experimental_rt_data[
            experimental_rt_data["stim_type"] == "positive"
        ]

        # 为每个等级创建条形
        for i, level in enumerate(level_order):
            # 对照组积极数据
            control_pos_level = control_pos_data[
                control_pos_data["intensity_level"] == level
            ]
            if not control_pos_level.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=control_pos_level["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=control_pos_level["std_rt"].values,
                            visible=True,
                        ),
                        name=f"{control_name}积极",
                        marker_color=control_color,
                        marker=dict(opacity=0.8),
                        legendgroup=f"{control_name}-positive",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=3,
                    col=1,
                )

            # 实验组积极数据
            experimental_pos_level = experimental_pos_data[
                experimental_pos_data["intensity_level"] == level
            ]
            if not experimental_pos_level.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=experimental_pos_level["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=experimental_pos_level["std_rt"].values,
                            visible=True,
                        ),
                        name=f"{experimental_name}积极",
                        marker_color=experimental_color,
                        marker=dict(opacity=0.8),
                        legendgroup=f"{experimental_name}-positive",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=3,
                    col=1,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(3)),
            ticktext=["模糊", "中等", "清晰"],
            row=3,
            col=1,
        )
        fig.update_yaxes(title_text="反应时(秒)", row=3, col=1)
        fig.update_xaxes(title_text="模糊等级", row=3, col=1)

    # 图8: 模糊等级对应反应时（消极）
    if (
        "rt_by_level_emotion" in control_full_metrics
        and "rt_by_level_emotion" in experimental_full_metrics
    ):
        control_rt_data = control_full_metrics["rt_by_level_emotion"].to_pandas()
        experimental_rt_data = experimental_full_metrics[
            "rt_by_level_emotion"
        ].to_pandas()

        # 只提取消极数据
        control_neg_data = control_rt_data[control_rt_data["stim_type"] == "negative"]
        experimental_neg_data = experimental_rt_data[
            experimental_rt_data["stim_type"] == "negative"
        ]

        # 为每个等级创建条形
        for i, level in enumerate(level_order):
            # 对照组消极数据
            control_neg_level = control_neg_data[
                control_neg_data["intensity_level"] == level
            ]
            if not control_neg_level.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=control_neg_level["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=control_neg_level["std_rt"].values,
                            visible=True,
                        ),
                        name=f"{control_name}消极",
                        marker_color=control_color,
                        marker=dict(opacity=0.8, pattern=control_negative_pattern),
                        legendgroup=f"{control_name}-negative",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=3,
                    col=2,
                )

            # 实验组消极数据
            experimental_neg_level = experimental_neg_data[
                experimental_neg_data["intensity_level"] == level
            ]
            if not experimental_neg_level.empty:
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=experimental_neg_level["mean_rt"].values,
                        error_y=dict(
                            type="data",
                            array=experimental_neg_level["std_rt"].values,
                            visible=True,
                        ),
                        name=f"{experimental_name}消极",
                        marker_color=experimental_color,
                        marker=dict(opacity=0.8, pattern=experimental_negative_pattern),
                        legendgroup=f"{experimental_name}-negative",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=3,
                    col=2,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(3)),
            ticktext=["模糊", "中等", "清晰"],
            row=3,
            col=2,
        )
        fig.update_yaxes(title_text="反应时(秒)", row=3, col=2)
        fig.update_xaxes(title_text="模糊等级", row=3, col=2)

    # 图9: 中性判定强度（积极、消极和中性分开，两组对比）
    if (
        "neutral_by_label_emotion" in control_full_metrics
        and "neutral_by_label_emotion" in experimental_full_metrics
    ):
        control_neutral_data = control_full_metrics[
            "neutral_by_label_emotion"
        ].to_pandas()
        experimental_neutral_data = experimental_full_metrics[
            "neutral_by_label_emotion"
        ].to_pandas()

        # 分离积极和消极数据
        control_pos_data = control_neutral_data[
            control_neutral_data["stim_type"] == "positive"
        ]
        control_neg_data = control_neutral_data[
            control_neutral_data["stim_type"] == "negative"
        ]
        experimental_pos_data = experimental_neutral_data[
            experimental_neutral_data["stim_type"] == "positive"
        ]
        experimental_neg_data = experimental_neutral_data[
            experimental_neutral_data["stim_type"] == "negative"
        ]

        # 处理消极数据（将x轴变为负数）
        control_neg_data_sorted = control_neg_data.copy()
        control_neg_data_sorted["label_intensity_signed"] = -control_neg_data_sorted[
            "label_intensity_signed"
        ]
        control_neg_data_sorted = control_neg_data_sorted.sort_values(
            "label_intensity_signed"
        )

        experimental_neg_data_sorted = experimental_neg_data.copy()
        experimental_neg_data_sorted[
            "label_intensity_signed"
        ] = -experimental_neg_data_sorted["label_intensity_signed"]
        experimental_neg_data_sorted = experimental_neg_data_sorted.sort_values(
            "label_intensity_signed"
        )

        # 处理积极数据
        control_pos_data_sorted = control_pos_data.sort_values("label_intensity_signed")
        experimental_pos_data_sorted = experimental_pos_data.sort_values(
            "label_intensity_signed"
        )

        # 添加中性刺激数据
        control_neutral_stim = control_all_trials.filter(
            pl.col("stim_type") == "neutral"
        )
        experimental_neutral_stim = experimental_all_trials.filter(
            pl.col("stim_type") == "neutral"
        )

        control_neutral_proportion = None
        experimental_neutral_proportion = None

        if control_neutral_stim.height > 0:
            control_neutral_choice_count = control_neutral_stim.filter(
                pl.col("choice_type") == "neutral"
            ).height
            total_control_neutral = control_neutral_stim.height
            if total_control_neutral > 0:
                control_neutral_proportion = (
                    control_neutral_choice_count / total_control_neutral
                )

        if experimental_neutral_stim.height > 0:
            experimental_neutral_choice_count = experimental_neutral_stim.filter(
                pl.col("choice_type") == "neutral"
            ).height
            total_experimental_neutral = experimental_neutral_stim.height
            if total_experimental_neutral > 0:
                experimental_neutral_proportion = (
                    experimental_neutral_choice_count / total_experimental_neutral
                )

        # 绘制对照组数据 - 使用lines+markers模式，所有点使用相同的marker
        if not control_pos_data_sorted.empty and not control_neg_data_sorted.empty:
            # 合并所有x和y数据
            control_x_all = (
                control_pos_data_sorted["label_intensity_signed"].tolist()
                + control_neg_data_sorted["label_intensity_signed"].tolist()
            )
            control_y_all = (
                control_pos_data_sorted["neutral_proportion"].tolist()
                + control_neg_data_sorted["neutral_proportion"].tolist()
            )

            if control_neutral_proportion is not None:
                control_x_all.append(0)
                control_y_all.append(control_neutral_proportion)

            # 按x值排序
            control_combined = list(zip(control_x_all, control_y_all))
            control_combined.sort(key=lambda x: x[0])
            control_x_sorted = [item[0] for item in control_combined]
            control_y_sorted = [item[1] for item in control_combined]

            # 绘制对照组线条和点（所有点使用相同的marker）
            fig.add_trace(
                go.Scatter(
                    x=control_x_sorted,
                    y=control_y_sorted,
                    mode="lines+markers",
                    line=dict(width=3, color=control_color, dash="solid"),
                    marker=dict(size=8, color=control_color, symbol="circle"),
                    name=f"{control_name}",
                    showlegend=True,
                ),
                row=4,
                col=1,
            )

        # 绘制实验组数据 - 使用lines+markers模式，所有点使用相同的marker
        if (
            not experimental_pos_data_sorted.empty
            and not experimental_neg_data_sorted.empty
        ):
            # 合并所有x和y数据
            experimental_x_all = (
                experimental_pos_data_sorted["label_intensity_signed"].tolist()
                + experimental_neg_data_sorted["label_intensity_signed"].tolist()
            )
            experimental_y_all = (
                experimental_pos_data_sorted["neutral_proportion"].tolist()
                + experimental_neg_data_sorted["neutral_proportion"].tolist()
            )

            if experimental_neutral_proportion is not None:
                experimental_x_all.append(0)
                experimental_y_all.append(experimental_neutral_proportion)

            # 按x值排序
            experimental_combined = list(zip(experimental_x_all, experimental_y_all))
            experimental_combined.sort(key=lambda x: x[0])
            experimental_x_sorted = [item[0] for item in experimental_combined]
            experimental_y_sorted = [item[1] for item in experimental_combined]

            # 绘制实验组线条和点（所有点使用相同的marker）
            fig.add_trace(
                go.Scatter(
                    x=experimental_x_sorted,
                    y=experimental_y_sorted,
                    mode="lines+markers",
                    line=dict(width=3, color=experimental_color, dash="solid"),
                    marker=dict(size=8, color=experimental_color, symbol="circle"),
                    name=f"{experimental_name}",
                    showlegend=True,
                ),
                row=4,
                col=1,
            )

        fig.update_yaxes(title_text="中性判定比例", range=[0, 1], row=4, col=1)
        fig.update_xaxes(
            title_text="标签强度（积极为正，消极为负，中性为0）", row=4, col=1
        )

    # 图10: 强度一致性分析（两组对比）
    if (
        "intensity_by_label_emotion" in control_full_metrics
        and "intensity_by_label_emotion" in experimental_full_metrics
    ):
        control_intensity_data = control_full_metrics[
            "intensity_by_label_emotion"
        ].to_pandas()
        experimental_intensity_data = experimental_full_metrics[
            "intensity_by_label_emotion"
        ].to_pandas()

        # 分离积极和消极数据
        control_pos_data = control_intensity_data[
            control_intensity_data["stim_type"] == "positive"
        ]
        control_neg_data = control_intensity_data[
            control_intensity_data["stim_type"] == "negative"
        ]
        experimental_pos_data = experimental_intensity_data[
            experimental_intensity_data["stim_type"] == "positive"
        ]
        experimental_neg_data = experimental_intensity_data[
            experimental_intensity_data["stim_type"] == "negative"
        ]

        # 处理中性刺激数据
        control_neutral_stim = control_all_trials.filter(
            pl.col("stim_type") == "neutral"
        )
        experimental_neutral_stim = experimental_all_trials.filter(
            pl.col("stim_type") == "neutral"
        )

        control_neutral_intensity = None
        control_neutral_std = None
        experimental_neutral_intensity = None
        experimental_neutral_std = None

        if control_neutral_stim.height > 0:
            correct_control_neutral = control_neutral_stim.filter(pl.col("correct"))
            if correct_control_neutral.height > 0:
                control_neutral_intensity = correct_control_neutral["intensity"].mean()
                control_neutral_std = correct_control_neutral["intensity"].std()

        if experimental_neutral_stim.height > 0:
            correct_experimental_neutral = experimental_neutral_stim.filter(
                pl.col("correct")
            )
            if correct_experimental_neutral.height > 0:
                experimental_neutral_intensity = correct_experimental_neutral[
                    "intensity"
                ].mean()
                experimental_neutral_std = correct_experimental_neutral[
                    "intensity"
                ].std()

        # 合并对照组所有数据（积极、消极和中性）
        if (
            not control_pos_data.empty
            and not control_neg_data.empty
            and control_neutral_intensity is not None
        ):
            control_pos_data = control_pos_data.sort_values("label_intensity_signed")
            control_pos_upper = (
                control_pos_data["mean_intensity"] + control_pos_data["std_intensity"]
            )
            control_pos_lower = (
                control_pos_data["mean_intensity"] - control_pos_data["std_intensity"]
            )
            control_pos_main = control_pos_data["mean_intensity"]

            # 处理消极数据（将x轴变为负数，y轴保持不变）
            control_neg_sorted = control_neg_data.sort_values("label_intensity_signed")
            control_neg_sorted = control_neg_sorted.copy()
            control_neg_sorted["label_intensity_signed"] = -control_neg_sorted[
                "label_intensity_signed"
            ]
            control_neg_sorted["mean_intensity"] = -control_neg_sorted["mean_intensity"]
            control_neg_sorted["std_intensity"] = -control_neg_sorted["std_intensity"]

            control_neg_upper = (
                control_neg_sorted["mean_intensity"]
                + control_neg_sorted["std_intensity"]
            )
            control_neg_lower = (
                control_neg_sorted["mean_intensity"]
                - control_neg_sorted["std_intensity"]
            )
            control_neg_main = control_neg_sorted["mean_intensity"]

            # 合并所有数据点
            control_x_raw = (
                control_pos_data["label_intensity_signed"].tolist()
                + control_neg_sorted["label_intensity_signed"].tolist()
                + [0]
            )
            control_y_raw = (
                control_pos_main.tolist()
                + control_neg_main.tolist()
                + [control_neutral_intensity]
            )
            control_y_raw_upper = (
                control_pos_upper.tolist()
                + control_neg_upper.tolist()
                + [control_neutral_intensity + control_neutral_std]
            )
            control_y_raw_lower = (
                control_pos_lower.tolist()
                + control_neg_lower.tolist()
                + [control_neutral_intensity - control_neutral_std]
            )

            # 按x值排序
            control_indices = list(range(len(control_x_raw)))
            control_indices.sort(key=lambda x: control_x_raw[x])

            control_x = [control_x_raw[i] for i in control_indices]
            control_y = [control_y_raw[i] for i in control_indices]
            control_y_upper = [control_y_raw_upper[i] for i in control_indices]
            control_y_lower = [control_y_raw_lower[i] for i in control_indices]

            # 绘制对照组置信区间和主线条
            control_lower, control_upper, control_main = draw_ci_scatter(
                x=control_x,
                y=control_y,
                y_upper=control_y_upper,
                y_lower=control_y_lower,
                name=f"{control_name}选择强度",
                color=control_color,
            )

            fig.add_trace(control_lower, row=4, col=2)
            fig.add_trace(control_upper, row=4, col=2)
            fig.add_trace(control_main, row=4, col=2)

        # 合并实验组所有数据（积极、消极和中性）
        if (
            not experimental_pos_data.empty
            and not experimental_neg_data.empty
            and experimental_neutral_intensity is not None
        ):
            experimental_pos_data = experimental_pos_data.sort_values(
                "label_intensity_signed"
            )
            experimental_pos_upper = (
                experimental_pos_data["mean_intensity"]
                + experimental_pos_data["std_intensity"]
            )
            experimental_pos_lower = (
                experimental_pos_data["mean_intensity"]
                - experimental_pos_data["std_intensity"]
            )
            experimental_pos_main = experimental_pos_data["mean_intensity"]

            # 处理消极数据（将x轴变为负数，y轴保持不变）
            experimental_neg_sorted = experimental_neg_data.sort_values(
                "label_intensity_signed"
            )
            experimental_neg_sorted = experimental_neg_sorted.copy()
            experimental_neg_sorted[
                "label_intensity_signed"
            ] = -experimental_neg_sorted["label_intensity_signed"]
            experimental_neg_sorted["mean_intensity"] = -experimental_neg_sorted[
                "mean_intensity"
            ]
            experimental_neg_sorted["std_intensity"] = -experimental_neg_sorted[
                "std_intensity"
            ]

            experimental_neg_upper = (
                experimental_neg_sorted["mean_intensity"]
                + experimental_neg_sorted["std_intensity"]
            )
            experimental_neg_lower = (
                experimental_neg_sorted["mean_intensity"]
                - experimental_neg_sorted["std_intensity"]
            )
            experimental_neg_main = experimental_neg_sorted["mean_intensity"]

            # 合并所有数据点
            experimental_x_raw = (
                experimental_pos_data["label_intensity_signed"].tolist()
                + experimental_neg_sorted["label_intensity_signed"].tolist()
                + [0]
            )
            experimental_y_raw = (
                experimental_pos_main.tolist()
                + experimental_neg_main.tolist()
                + [experimental_neutral_intensity]
            )
            experimental_y_raw_upper = (
                experimental_pos_upper.tolist()
                + experimental_neg_upper.tolist()
                + [experimental_neutral_intensity + experimental_neutral_std]
            )
            experimental_y_raw_lower = (
                experimental_pos_lower.tolist()
                + experimental_neg_lower.tolist()
                + [experimental_neutral_intensity - experimental_neutral_std]
            )

            # 按x值排序
            experimental_indices = list(range(len(experimental_x_raw)))
            experimental_indices.sort(key=lambda x: experimental_x_raw[x])

            experimental_x = [experimental_x_raw[i] for i in experimental_indices]
            experimental_y = [experimental_y_raw[i] for i in experimental_indices]
            experimental_y_upper = [
                experimental_y_raw_upper[i] for i in experimental_indices
            ]
            experimental_y_lower = [
                experimental_y_raw_lower[i] for i in experimental_indices
            ]

            # 绘制实验组置信区间和主线条
            experimental_lower, experimental_upper, experimental_main = draw_ci_scatter(
                x=experimental_x,
                y=experimental_y,
                y_upper=experimental_y_upper,
                y_lower=experimental_y_lower,
                name=f"{experimental_name}选择强度",
                color=experimental_color,
            )

            fig.add_trace(experimental_lower, row=4, col=2)
            fig.add_trace(experimental_upper, row=4, col=2)
            fig.add_trace(experimental_main, row=4, col=2)

        # 添加对角线（理想一致性线）
        max_val = 9
        fig.add_trace(
            go.Scatter(
                x=[-max_val, max_val],
                y=[-max_val, max_val],
                mode="lines",
                line=dict(dash="dash", color="gray", width=2),
                name="理想一致性线",
                showlegend=True,
            ),
            row=4,
            col=2,
        )

        fig.update_yaxes(
            title_text="选择强度", range=[-max_val - 1, max_val + 1], row=4, col=2
        )
        fig.update_xaxes(
            title_text="标签强度（积极为正，消极为负，中性为0）", row=4, col=2
        )
    # 图11: 各标签强度对应选择强度（总体）- 两组对比
    if (
        "intensity_by_label_emotion" in control_full_metrics
        and "intensity_by_label_emotion" in experimental_full_metrics
    ):
        control_intensity_data = control_full_metrics[
            "intensity_by_label_emotion"
        ].to_pandas()
        experimental_intensity_data = experimental_full_metrics[
            "intensity_by_label_emotion"
        ].to_pandas()

        # 合并积极和消极数据（不分情绪类型）
        control_all_data = control_intensity_data.copy()
        experimental_all_data = experimental_intensity_data.copy()

        # 获取所有强度级别
        control_intensities = control_intensity_data["label_intensity_signed"].unique()
        experimental_intensities = experimental_intensity_data[
            "label_intensity_signed"
        ].unique()
        all_intensities = sorted(
            set(np.concatenate([control_intensities, experimental_intensities]))
        )

        # 计算每个强度级别的总体平均值
        control_overall_means = []
        control_overall_stds = []
        experimental_overall_means = []
        experimental_overall_stds = []

        for intensity in all_intensities:
            # 对照组总体平均值（积极和消极合并）
            control_intensity_rows = control_all_data[
                control_all_data["label_intensity_signed"] == intensity
            ]
            if len(control_intensity_rows) > 0:
                # 加权平均：根据试次数加权
                total_trials = control_intensity_rows["n_trials"].sum()
                if total_trials > 0:
                    weighted_mean = (
                        control_intensity_rows["mean_intensity"]
                        * control_intensity_rows["n_trials"]
                    ).sum() / total_trials
                    # 计算合并标准差
                    weighted_var = (
                        (control_intensity_rows["std_intensity"] ** 2)
                        * (control_intensity_rows["n_trials"] - 1)
                    ).sum()
                    weighted_var = weighted_var / (
                        total_trials - len(control_intensity_rows)
                    )
                    weighted_std = np.sqrt(weighted_var)
                else:
                    weighted_mean = control_intensity_rows["mean_intensity"].mean()
                    weighted_std = control_intensity_rows["std_intensity"].mean()
            else:
                weighted_mean = np.nan
                weighted_std = np.nan

            control_overall_means.append(weighted_mean)
            control_overall_stds.append(weighted_std)

            # 实验组总体平均值（积极和消极合并）
            experimental_intensity_rows = experimental_all_data[
                experimental_all_data["label_intensity_signed"] == intensity
            ]
            if len(experimental_intensity_rows) > 0:
                total_trials = experimental_intensity_rows["n_trials"].sum()
                if total_trials > 0:
                    weighted_mean = (
                        experimental_intensity_rows["mean_intensity"]
                        * experimental_intensity_rows["n_trials"]
                    ).sum() / total_trials
                    # 计算合并标准差
                    weighted_var = (
                        (experimental_intensity_rows["std_intensity"] ** 2)
                        * (experimental_intensity_rows["n_trials"] - 1)
                    ).sum()
                    weighted_var = weighted_var / (
                        total_trials - len(experimental_intensity_rows)
                    )
                    weighted_std = np.sqrt(weighted_var)
                else:
                    weighted_mean = experimental_intensity_rows["mean_intensity"].mean()
                    weighted_std = experimental_intensity_rows["std_intensity"].mean()
            else:
                weighted_mean = np.nan
                weighted_std = np.nan

            experimental_overall_means.append(weighted_mean)
            experimental_overall_stds.append(weighted_std)

        # 为每个强度级别创建条形
        for i, intensity in enumerate(all_intensities):
            # 对照组总体数据
            if not np.isnan(control_overall_means[i]):
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=[control_overall_means[i]],
                        error_y=dict(
                            type="data",
                            array=[control_overall_stds[i]],
                            visible=True,
                        ),
                        name=f"{control_name}总体"
                        if intensity == all_intensities[0]
                        else "",
                        marker_color=control_color,
                        marker=dict(opacity=0.6),
                        legendgroup=f"{control_name}-overall",
                        showlegend=bool(intensity == all_intensities[0]),
                        width=0.4,
                    ),
                    row=5,
                    col=1,
                )

            # 实验组总体数据
            if not np.isnan(experimental_overall_means[i]):
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=[experimental_overall_means[i]],
                        error_y=dict(
                            type="data",
                            array=[experimental_overall_stds[i]],
                            visible=True,
                        ),
                        name=f"{experimental_name}总体"
                        if intensity == all_intensities[0]
                        else "",
                        marker_color=experimental_color,
                        marker=dict(opacity=0.6),
                        legendgroup=f"{experimental_name}-overall",
                        showlegend=bool(intensity == all_intensities[0]),
                        width=0.4,
                    ),
                    row=5,
                    col=1,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(all_intensities))),
            ticktext=[str(i) for i in all_intensities],
            row=5,
            col=1,
        )
        fig.update_yaxes(title_text="选择强度", range=[0, 10], row=5, col=1)
        fig.update_xaxes(title_text="标签强度", row=5, col=1)

    # 图12: 各标签强度对应反应时（总体）- 两组对比
    if (
        "rt_by_label_emotion" in control_full_metrics
        and "rt_by_label_emotion" in experimental_full_metrics
    ):
        control_rt_data = control_full_metrics["rt_by_label_emotion"].to_pandas()
        experimental_rt_data = experimental_full_metrics[
            "rt_by_label_emotion"
        ].to_pandas()

        # 合并积极和消极数据（不分情绪类型）
        control_all_rt = control_rt_data.copy()
        experimental_all_rt = experimental_rt_data.copy()

        # 获取所有强度级别
        control_intensities = control_rt_data["label_intensity_signed"].unique()
        experimental_intensities = experimental_rt_data[
            "label_intensity_signed"
        ].unique()
        all_intensities = sorted(
            set(np.concatenate([control_intensities, experimental_intensities]))
        )

        # 计算每个强度级别的总体平均值
        control_rt_means = []
        control_rt_stds = []
        experimental_rt_means = []
        experimental_rt_stds = []

        for intensity in all_intensities:
            # 对照组总体反应时平均值
            control_rt_rows = control_all_rt[
                control_all_rt["label_intensity_signed"] == intensity
            ]
            if len(control_rt_rows) > 0:
                total_trials = control_rt_rows["n_trials"].sum()
                if total_trials > 0:
                    weighted_mean = (
                        control_rt_rows["mean_rt"] * control_rt_rows["n_trials"]
                    ).sum() / total_trials
                    # 计算合并标准差
                    weighted_var = (
                        (control_rt_rows["std_rt"] ** 2)
                        * (control_rt_rows["n_trials"] - 1)
                    ).sum()
                    weighted_var = weighted_var / (total_trials - len(control_rt_rows))
                    weighted_std = np.sqrt(weighted_var)
                else:
                    weighted_mean = control_rt_rows["mean_rt"].mean()
                    weighted_std = control_rt_rows["std_rt"].mean()
            else:
                weighted_mean = np.nan
                weighted_std = np.nan

            control_rt_means.append(weighted_mean)
            control_rt_stds.append(weighted_std)

            # 实验组总体反应时平均值
            experimental_rt_rows = experimental_all_rt[
                experimental_all_rt["label_intensity_signed"] == intensity
            ]
            if len(experimental_rt_rows) > 0:
                total_trials = experimental_rt_rows["n_trials"].sum()
                if total_trials > 0:
                    weighted_mean = (
                        experimental_rt_rows["mean_rt"]
                        * experimental_rt_rows["n_trials"]
                    ).sum() / total_trials
                    # 计算合并标准差
                    weighted_var = (
                        (experimental_rt_rows["std_rt"] ** 2)
                        * (experimental_rt_rows["n_trials"] - 1)
                    ).sum()
                    weighted_var = weighted_var / (
                        total_trials - len(experimental_rt_rows)
                    )
                    weighted_std = np.sqrt(weighted_var)
                else:
                    weighted_mean = experimental_rt_rows["mean_rt"].mean()
                    weighted_std = experimental_rt_rows["std_rt"].mean()
            else:
                weighted_mean = np.nan
                weighted_std = np.nan

            experimental_rt_means.append(weighted_mean)
            experimental_rt_stds.append(weighted_std)

        # 为每个强度级别创建条形
        for i, intensity in enumerate(all_intensities):
            # 对照组总体反应时数据
            if not np.isnan(control_rt_means[i]):
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=[control_rt_means[i]],
                        error_y=dict(
                            type="data",
                            array=[control_rt_stds[i]],
                            visible=True,
                        ),
                        name=f"{control_name}总体",
                        marker_color=control_color,
                        marker=dict(opacity=0.6),
                        legendgroup=f"{control_name}-overall",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=5,
                    col=2,
                )

            # 实验组总体反应时数据
            if not np.isnan(experimental_rt_means[i]):
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=[experimental_rt_means[i]],
                        error_y=dict(
                            type="data",
                            array=[experimental_rt_stds[i]],
                            visible=True,
                        ),
                        name=f"{experimental_name}总体",
                        marker_color=experimental_color,
                        marker=dict(opacity=0.6),
                        legendgroup=f"{experimental_name}-overall",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=5,
                    col=2,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(len(all_intensities))),
            ticktext=[str(i) for i in all_intensities],
            row=5,
            col=2,
        )
        fig.update_yaxes(title_text="反应时(秒)", row=5, col=2)
        fig.update_xaxes(title_text="标签强度", row=5, col=2)

    # 图13: 模糊等级对应选择强度（总体）- 两组对比
    if (
        "intensity_by_level_emotion" in control_full_metrics
        and "intensity_by_level_emotion" in experimental_full_metrics
    ):
        control_intensity_data = control_full_metrics[
            "intensity_by_level_emotion"
        ].to_pandas()
        experimental_intensity_data = experimental_full_metrics[
            "intensity_by_level_emotion"
        ].to_pandas()

        # 合并积极和消极数据
        control_all_data = control_intensity_data.copy()
        experimental_all_data = experimental_intensity_data.copy()

        # 定义等级顺序
        level_order = ["blur", "mid", "clear"]

        # 计算每个模糊等级的总体平均值
        for i, level in enumerate(level_order):
            # 对照组该等级的数据
            control_level_rows = control_all_data[
                control_all_data["intensity_level"] == level
            ]
            if len(control_level_rows) > 0:
                total_trials = control_level_rows["n_trials"].sum()
                if total_trials > 0:
                    weighted_mean = (
                        control_level_rows["mean_intensity"]
                        * control_level_rows["n_trials"]
                    ).sum() / total_trials
                    # 计算合并标准差
                    weighted_var = (
                        (control_level_rows["std_intensity"] ** 2)
                        * (control_level_rows["n_trials"] - 1)
                    ).sum()
                    weighted_var = weighted_var / (
                        total_trials - len(control_level_rows)
                    )
                    weighted_std = np.sqrt(weighted_var)
                else:
                    weighted_mean = control_level_rows["mean_intensity"].mean()
                    weighted_std = control_level_rows["std_intensity"].mean()
            else:
                weighted_mean = np.nan
                weighted_std = np.nan

            # 实验组该等级的数据
            experimental_level_rows = experimental_all_data[
                experimental_all_data["intensity_level"] == level
            ]
            if len(experimental_level_rows) > 0:
                total_trials = experimental_level_rows["n_trials"].sum()
                if total_trials > 0:
                    weighted_mean_exp = (
                        experimental_level_rows["mean_intensity"]
                        * experimental_level_rows["n_trials"]
                    ).sum() / total_trials
                    # 计算合并标准差
                    weighted_var_exp = (
                        (experimental_level_rows["std_intensity"] ** 2)
                        * (experimental_level_rows["n_trials"] - 1)
                    ).sum()
                    weighted_var_exp = weighted_var_exp / (
                        total_trials - len(experimental_level_rows)
                    )
                    weighted_std_exp = np.sqrt(weighted_var_exp)
                else:
                    weighted_mean_exp = experimental_level_rows["mean_intensity"].mean()
                    weighted_std_exp = experimental_level_rows["std_intensity"].mean()
            else:
                weighted_mean_exp = np.nan
                weighted_std_exp = np.nan

            # 绘制对照组条形
            if not np.isnan(weighted_mean):
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=[weighted_mean],
                        error_y=dict(
                            type="data",
                            array=[weighted_std],
                            visible=True,
                        ),
                        name=f"{control_name}总体",
                        marker_color=control_color,
                        marker=dict(opacity=0.6),
                        legendgroup=f"{control_name}-overall",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=6,
                    col=1,
                )

            # 绘制实验组条形
            if not np.isnan(weighted_mean_exp):
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=[weighted_mean_exp],
                        error_y=dict(
                            type="data",
                            array=[weighted_std_exp],
                            visible=True,
                        ),
                        name=f"{experimental_name}总体",
                        marker_color=experimental_color,
                        marker=dict(opacity=0.6),
                        legendgroup=f"{experimental_name}-overall",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=6,
                    col=1,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(3)),
            ticktext=["模糊", "中等", "清晰"],
            row=6,
            col=1,
        )
        fig.update_yaxes(title_text="选择强度", range=[0, 10], row=6, col=1)
        fig.update_xaxes(title_text="模糊等级", row=6, col=1)

    # 图14: 模糊等级对应反应时（总体）- 两组对比
    if (
        "rt_by_level_emotion" in control_full_metrics
        and "rt_by_level_emotion" in experimental_full_metrics
    ):
        control_rt_data = control_full_metrics["rt_by_level_emotion"].to_pandas()
        experimental_rt_data = experimental_full_metrics[
            "rt_by_level_emotion"
        ].to_pandas()

        # 合并积极和消极数据
        control_all_rt = control_rt_data.copy()
        experimental_all_rt = experimental_rt_data.copy()

        # 定义等级顺序
        level_order = ["blur", "mid", "clear"]

        # 计算每个模糊等级的总体反应时平均值
        for i, level in enumerate(level_order):
            # 对照组该等级的反应时数据
            control_level_rows = control_all_rt[
                control_all_rt["intensity_level"] == level
            ]
            if len(control_level_rows) > 0:
                total_trials = control_level_rows["n_trials"].sum()
                if total_trials > 0:
                    weighted_mean = (
                        control_level_rows["mean_rt"] * control_level_rows["n_trials"]
                    ).sum() / total_trials
                    # 计算合并标准差
                    weighted_var = (
                        (control_level_rows["std_rt"] ** 2)
                        * (control_level_rows["n_trials"] - 1)
                    ).sum()
                    weighted_var = weighted_var / (
                        total_trials - len(control_level_rows)
                    )
                    weighted_std = np.sqrt(weighted_var)
                else:
                    weighted_mean = control_level_rows["mean_rt"].mean()
                    weighted_std = control_level_rows["std_rt"].mean()
            else:
                weighted_mean = np.nan
                weighted_std = np.nan

            # 实验组该等级的反应时数据
            experimental_level_rows = experimental_all_rt[
                experimental_all_rt["intensity_level"] == level
            ]
            if len(experimental_level_rows) > 0:
                total_trials = experimental_level_rows["n_trials"].sum()
                if total_trials > 0:
                    weighted_mean_exp = (
                        experimental_level_rows["mean_rt"]
                        * experimental_level_rows["n_trials"]
                    ).sum() / total_trials
                    # 计算合并标准差
                    weighted_var_exp = (
                        (experimental_level_rows["std_rt"] ** 2)
                        * (experimental_level_rows["n_trials"] - 1)
                    ).sum()
                    weighted_var_exp = weighted_var_exp / (
                        total_trials - len(experimental_level_rows)
                    )
                    weighted_std_exp = np.sqrt(weighted_var_exp)
                else:
                    weighted_mean_exp = experimental_level_rows["mean_rt"].mean()
                    weighted_std_exp = experimental_level_rows["std_rt"].mean()
            else:
                weighted_mean_exp = np.nan
                weighted_std_exp = np.nan

            # 绘制对照组条形
            if not np.isnan(weighted_mean):
                fig.add_trace(
                    go.Bar(
                        x=[i - 0.2],
                        y=[weighted_mean],
                        error_y=dict(
                            type="data",
                            array=[weighted_std],
                            visible=True,
                        ),
                        name=f"{control_name}总体",
                        marker_color=control_color,
                        marker=dict(opacity=0.6),
                        legendgroup=f"{control_name}-overall",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=6,
                    col=2,
                )

            # 绘制实验组条形
            if not np.isnan(weighted_mean_exp):
                fig.add_trace(
                    go.Bar(
                        x=[i + 0.2],
                        y=[weighted_mean_exp],
                        error_y=dict(
                            type="data",
                            array=[weighted_std_exp],
                            visible=True,
                        ),
                        name=f"{experimental_name}总体",
                        marker_color=experimental_color,
                        marker=dict(opacity=0.6),
                        legendgroup=f"{experimental_name}-overall",
                        showlegend=False,
                        width=0.4,
                    ),
                    row=6,
                    col=2,
                )

        # 设置x轴刻度
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(3)),
            ticktext=["模糊", "中等", "清晰"],
            row=6,
            col=2,
        )
        fig.update_yaxes(title_text="反应时(秒)", row=6, col=2)
        fig.update_xaxes(title_text="模糊等级", row=6, col=2)

    # 调整布局高度以适应新增的行
    fig.update_layout(
        height=400 * 6,  # 增加高度以适应6行
        width=1800,
        title=dict(
            text=f"面部情绪识别{control_name} vs {experimental_name}组间对比分析",
            font=dict(size=24, family="Arial Black"),
            x=0.5,
        ),
        showlegend=True,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        barmode="group",
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
    """组面部情绪识别分析 - 使用新指标"""
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
        if "intensity" in metric and "neutral" not in metric:
            ref_value = 5.0  # 中等强度
        elif "rt" in metric:
            ref_value = 1.0  # 1秒反应时
        elif "neutral_intensity" in metric:
            ref_value = 0.1  # 10%的中性判定比例
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

    fig_spec = create_single_group_visualizations(
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
    """比较对照组和实验组 - 使用新指标进行比较"""

    control_name = groups[0] if groups else "control"

    control_group_results = run_group_emotion_analysis(
        control_files, result_dir, control_name
    )
    control_results = control_group_results["all_results"]
    control_comprehensive_metrics = control_group_results["comprehensive_metrics"]
    control_trials = control_group_results.get("all_trials", [])

    experimental_name = groups[1] if groups and len(groups) > 1 else "experimental"

    experimental_group_results = run_group_emotion_analysis(
        experimental_files, result_dir, experimental_name
    )
    experimental_results = experimental_group_results["all_results"]
    experimental_comprehensive_metrics = experimental_group_results[
        "comprehensive_metrics"
    ]
    experimental_trials = experimental_group_results.get("all_trials", [])

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

    # 使用新指标创建可视化
    fig_spec = create_multi_group_visualizations(
        control_metrics=control_comprehensive_metrics,
        experimental_metrics=experimental_comprehensive_metrics,
        control_name=control_name,
        experimental_name=experimental_name,
        control_trials=control_trials,
        experimental_trials=experimental_trials,
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
