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

key_metrics = [
    "overall_accuracy",
    "positive_accuracy",
    "negative_accuracy",
    "neutral_accuracy",
    "positive_low_rt",
    "positive_mid_rt",
    "positive_high_rt",
    "negative_low_rt",
    "negative_mid_rt",
    "negative_high_rt",
    "positive_low_acc",
    "positive_mid_acc",
    "positive_high_acc",
    "negative_low_acc",
    "negative_mid_acc",
    "negative_high_acc",
]

metric_names = [
    "总体正确率",
    "积极正确率",
    "消极正确率",
    "中性正确率",
    "低积极反应时",
    "中积极反应时",
    "高积极反应时",
    "低消极反应时",
    "中消极反应时",
    "高消极反应时",
    "低积极正确率",
    "中积极正确率",
    "高积极正确率",
    "低消极正确率",
    "中消极正确率",
    "高消极正确率",
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

        # [ ] 去除过大过小而不是强制转换
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

    return metrics


def calculate_intensity_metrics(trials_df: pl.DataFrame) -> dict[str, Any]:
    """计算强度指标和中性阈值"""
    metrics = {}

    # 强度一致性分析（非中性刺激）
    df_intensity = trials_df.filter(pl.col("stim_type") != "neutral").with_columns(
        [
            (pl.col("intensity") - pl.col("label_intensity")).alias("intensity_diff"),
            (pl.col("intensity") - pl.col("label_intensity"))
            .abs()
            .alias("abs_intensity_diff"),
        ]
    )

    if df_intensity.height > 0:
        intensity_stats = df_intensity.group_by("stim_type").agg(
            [
                pl.col("intensity_diff").mean().alias("mean_diff"),
                pl.col("intensity_diff").std().alias("std_diff"),
                pl.col("abs_intensity_diff").mean().alias("mean_abs_diff"),
                pl.col("label_intensity").mean().alias("mean_label_intensity"),
            ]
        )
        metrics["intensity_stats"] = intensity_stats

        # 强度相关性分析
        intensity_corr_data = df_intensity.filter(
            pl.col("label_intensity").is_not_null() & pl.col("intensity").is_not_null()
        )

        if intensity_corr_data.height >= 3:
            corr_df = intensity_corr_data.select(
                ["label_intensity", "intensity"]
            ).to_pandas()
            corr, p_val = stats.pearsonr(
                corr_df["label_intensity"], corr_df["intensity"]
            )
            metrics["intensity_correlation"] = {
                "r": corr,
                "p": p_val,
                "n": len(corr_df),
            }

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


def create_visualizations(
    trials_df: pl.DataFrame,
    metrics: dict[str, Any],
    stats_results: dict[str, Any],
    key_metrics: dict[str, float],
    result_dir: Path,
) -> list[go.Figure]:
    """可视化图表"""
    figs = []
    emotion_correct_pd = metrics["emotion_accuracy"].to_pandas()
    df_pd = trials_df.to_pandas()

    df_intensity = trials_df.filter(pl.col("stim_type") != "neutral")
    df_intensity_pd = df_intensity.to_pandas() if df_intensity.height > 0 else None

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "1. 情绪类型正确率对比",
            "2. 不同情绪类型的反应时分布",
            "3. 反应时与正确率的关系",
            "4. 强度评分一致性",
            "5. 中性阈值分析",
            "6. 分块正确率变化",
            "7. 关键指标总结",
            "8. 速度-准确性权衡",
        ),
        specs=[
            [{"type": "bar"}, {"type": "box"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
            [{"type": "table"}, None, {"type": "scatter"}],
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
    scatter_sample = df_pd.sample(frac=0.3, random_state=42)
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

    # 图4: 强度评分一致性
    if df_intensity_pd is not None and len(df_intensity_pd) > 0:
        fig.add_trace(
            go.Scatter(
                x=df_intensity_pd["label_intensity"],
                y=df_intensity_pd["intensity"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=df_intensity_pd["stim_type"].map(
                        {"positive": 0, "negative": 1}
                    ),
                    colorscale=["#00cc96", "#ef553b"],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="情绪类型", side="top"),
                        tickvals=[0, 1],
                        ticktext=["积极", "消极"],
                        len=0.1,
                        y=0.5,
                        x=1.1,
                        thickness=15,
                        orientation="h",
                    ),
                ),
                text=df_intensity_pd["stim_type"],
                hovertemplate="<b>标签强度</b>: %{x}<br><b>被试评分</b>: %{y}<br><b>类型</b>: %{text}<extra></extra>",
                name="强度一致性",
            ),
            row=2,
            col=1,
        )
        max_val = max(df_intensity_pd[["label_intensity", "intensity"]].max().max(), 9)
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                showlegend=False,
                name="对角线",
            ),
            row=2,
            col=1,
        )
    fig.update_xaxes(title_text="标签强度 (预设)", row=2, col=1)
    fig.update_yaxes(title_text="被试评分强度", row=2, col=1)

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

    # 图6: 分块正确率变化
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
        col=3,
    )
    fig.update_yaxes(range=[0, 1.05], title_text="正确率", row=2, col=3)
    fig.update_xaxes(title_text="区块编号", row=2, col=3)

    # 图7: 关键指标总结表格
    metrics_table = go.Table(
        header=dict(
            values=["指标", "值", "解释"], fill_color="lightblue", align="left"
        ),
        cells=dict(
            values=[
                [
                    "总体正确率",
                    "中位反应时",
                    "积极情绪正确率",
                    "消极情绪正确率",
                    "中性情绪正确率",
                    "总试次数",
                ],
                [
                    f"{key_metrics.get('overall_accuracy', 0):.3f}",
                    f"{key_metrics.get('median_rt', 0):.3f}",
                    f"{key_metrics.get('positive_accuracy', 0):.3f}",
                    f"{key_metrics.get('negative_accuracy', 0):.3f}",
                    f"{key_metrics.get('neutral_accuracy', 0):.3f}",
                    f"{key_metrics.get('total_trials', 0)}",
                ],
                [
                    "整体表现",
                    "反应速度",
                    "积极情绪识别",
                    "消极情绪识别",
                    "中性情绪识别",
                    "数据量",
                ],
            ]
        ),
    )
    fig.add_trace(metrics_table, row=3, col=1)

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

    if "neutral_threshold_by_stimulus" in metrics:
        create_neutral_threshold_visualization(metrics, result_dir)

    figs.append(fig)
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


def calculate_key_metrics(
    trials_df: pl.DataFrame, metrics: dict[str, Any]
) -> dict[str, float]:
    """计算关键指标"""
    key_metrics = {
        "overall_accuracy": float(metrics["overall_accuracy"]),
        "median_rt": float(metrics["median_rt"]),
        "total_trials": int(metrics["total_trials"]),
    }

    emotion_acc = metrics["emotion_accuracy"]
    for stim_type in ["positive", "negative", "neutral"]:
        stim_data = emotion_acc.filter(pl.col("stim_type") == stim_type)
        if stim_data.height > 0:
            key_metrics[f"{stim_type}_accuracy"] = float(stim_data["correct_rate"][0])

    rt_data = metrics["reaction_time"]
    if rt_data.height >= 2:
        positive_rt = rt_data.filter(pl.col("stim_type") == "positive")
        negative_rt = rt_data.filter(pl.col("stim_type") == "negative")
        if positive_rt.height > 0 and negative_rt.height > 0:
            key_metrics["rt_pos_neg_diff"] = float(
                positive_rt["mean_rt"][0] - negative_rt["mean_rt"][0]
            )

    # 强度相关性
    if "intensity_correlation" in metrics:
        key_metrics["intensity_correlation_r"] = metrics["intensity_correlation"]["r"]

    return key_metrics


def save_results(
    metrics: dict[str, Any],
    stats_results: dict[str, Any],
    key_metrics: dict[str, float],
    result_dir: Path,
) -> dict[str, Path]:
    """保存结果"""
    saved_files = {}

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

    if "intensity_stats" in metrics:
        intensity_df = metrics["intensity_stats"].to_pandas()
        intensity_path = result_dir / "emotion_face_intensity_analysis.csv"
        intensity_df.to_csv(intensity_path, index=False)
        saved_files["intensity_analysis"] = intensity_path

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
    key_metrics = calculate_key_metrics(trials_df, metrics)

    # 6. 创建可视化
    fig = create_visualizations(  # noqa: F841
        trials_df, metrics, stats_results, key_metrics, result_dir
    )

    # 7. 保存结果
    saved_files = save_results(metrics, stats_results, key_metrics, result_dir)

    # 8. 生成报告
    report = {
        "data_summary": {
            "total_trials": trials_df.height,
            "num_blocks": len(trials_df["block_index"].unique()),
            "overall_accuracy": float(metrics["overall_accuracy"]),
            "median_rt": float(metrics["median_rt"]),
        },
        "key_metrics": key_metrics,
        "metrics": metrics,
        "statistical_results": stats_results,
        "saved_files": saved_files,
    }

    print(f"\n✅ 分析完成！结果保存在: {result_dir}")
    return report


def create_group_comparison_visualizations_single_group(
    group_metrics: list[dict[str, float]],
) -> list[go.Figure]:
    """单个组的组分析可视化"""
    # [ ], 这里的参数记得改
    figs = []
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "1. 各被试总体正确率分布",
            "3. 描述性统计",
            # "6. 指标分布箱形图",
        ),
        specs=[[{"type": "bar"}], [{"type": "table"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # 图1: 各被试总体正确率分布
    accuracy_values = [m["overall_accuracy"] for m in group_metrics]
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

    metrics_df = pd.DataFrame(group_metrics)

    # 图3: 描述性统计表格
    descriptive_stats = []
    for metric, name in zip(key_metrics, metric_names):
        if metric in metrics_df.columns:
            values = metrics_df[metric].dropna()
            if len(values) > 0:
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
            col=1,
        )

    # 图6: 指标分布箱形图
    # for i, (metric, name) in enumerate(zip(key_metrics_list, metric_names)):
    #     if metric in metrics_df.columns:
    #         values = [m[metric] for m in group_metrics if metric in m]
    #         if values:
    #             fig.add_trace(
    #                 go.Box(
    #                     y=values,
    #                     name=name,
    #                     boxpoints="all",
    #                     jitter=0.3,
    #                     pointpos=-1.8,
    #                     marker_color="lightblue",
    #                     showlegend=False,
    #                 ),
    #                 row=2,
    #                 col=3,
    #             )

    fig.update_layout(
        title=dict(
            text="面部情绪识别组分析报告",
            font=dict(size=22, family="Arial Black"),
            x=0.5,
        ),
        showlegend=True,
        template="plotly_white",
    )

    # fig.write_html(str(result_dir / "emotion_face_group_analysis_report.html"))

    figs.append(fig)
    return figs


def create_group_comparison_visualizations(
    control_metrics: list[dict[str, float]],
    experimental_metrics: list[dict[str, float]],
) -> list[go.Figure]:
    """创建组间比较可视化"""

    figs = []
    # 准备数据
    control_values = {}
    experimental_values = {}

    for metric in control_metrics[0].keys():
        control_values[metric] = [m[metric] for m in control_metrics if metric in m]
        experimental_values[metric] = [
            m[metric] for m in experimental_metrics if metric in m
        ]

    # 创建图表
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=("3. 关键指标对比",),
        specs=[[{"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.2,
    )

    # TODO: 这里实现
    # 图1-3: 指标分布箱形图

    # for i, (metric, name) in enumerate(zip(key_metrics, metric_names)):
    #     print(f"row: {i // 2 + 1}, col: {i % 2 + 1}")
    #     if metric in control_values and metric in experimental_values:
    #         fig.add_trace(
    #             go.Box(
    #                 y=control_values[metric],
    #                 name="对照组",
    #                 boxpoints="all",
    #                 jitter=0.3,
    #                 pointpos=-1.8,
    #                 marker_color="lightgreen",
    #                 showlegend=(i == 0),
    #             ),
    #             row=i // 2 + 1,
    #             col=i % 2 + 1,
    #         )

    #         fig.add_trace(
    #             go.Box(
    #                 y=experimental_values[metric],
    #                 name="实验组",
    #                 boxpoints="all",
    #                 jitter=0.3,
    #                 pointpos=-1.8,
    #                 marker_color="lightcoral",
    #                 showlegend=(i == 0),
    #             ),
    #             row=i // 2 + 1,
    #             col=i % 2 + 1,
    #         )

    # 图4: 关键指标对比
    control_means = []
    control_stds = []
    experimental_means = []
    experimental_stds = []
    valid_metrics = []
    valid_names = []

    for metric, name in zip(key_metrics, metric_names):
        if metric in control_values and metric in experimental_values:
            control_means.append(np.mean(control_values[metric]))
            control_stds.append(np.std(control_values[metric], ddof=1))
            experimental_means.append(np.mean(experimental_values[metric]))
            experimental_stds.append(np.std(experimental_values[metric], ddof=1))
            valid_metrics.append(metric)
            valid_names.append(name)

    if valid_metrics:
        x_positions = np.arange(len(valid_metrics))

        fig.add_trace(
            go.Bar(
                x=x_positions - 0.2,
                y=control_means,
                name="对照组",
                marker_color="green",
                error_y=dict(type="data", array=control_stds, visible=True),
                width=0.4,
            ),
            row=1,
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
            row=1,
            col=1,
        )

        fig.update_xaxes(ticktext=valid_names, tickvals=x_positions, row=1, col=1)

    fig.update_layout(
        title=dict(
            text="面部情绪识别组间比较分析报告",
            font=dict(size=22, family="Arial Black"),
            x=0.5,
        ),
        showlegend=True,
        template="plotly_white",
    )

    # fig.write_html(str(result_dir / "emotion_face_group_comparison_report.html"))

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
    """组面部情绪识别分析"""
    if result_dir is None:
        result_dir = Path("emotion_face_group_results")

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

            result = analyze_emotion_face_data(
                df=df,
                target_blocks=[0, 1],
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
        print("⚠️ 被试数量不足，无法进行统计检验")
        return {"all_results": all_results}

    # TODO: 没有参考值
    statistical_results = {}

    for metric in key_metrics:
        group_values = [m[metric] for m in group_metrics if metric in m]

        if len(group_values) < 2:
            statistical_results[metric] = {"error": "样本量不足"}
            continue

        # 单样本t检验：与0比较（对于正确率）或与0.5比较（对于反应时）
        if "accuracy" in metric or metric == "overall_accuracy":
            # 正确率：与1/3（随机水平）比较
            ref_value = 0.8
        elif "rt" in metric:
            # 反应时：与0.5比较
            ref_value = 0.5
        else:
            ref_value = 500

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
        if cohens_d is not None:
            sample_size_info = calculate_sample_size(
                effect_size=abs(cohens_d),
                alpha=0.05,
                power=0.8,
                test_type="one_sample",  # 假设未来进行两组比较
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

    if statistical_results:
        stats_test_df = pd.DataFrame(statistical_results).T
        stats_test_df.to_csv(result_dir / "group_statistical_tests.csv")

    fig_spec = create_group_comparison_visualizations_single_group(
        group_metrics,
    )

    fig_common = create_common_single_group_figures(
        group_metrics, statistical_results, key_metrics, metric_names
    )

    figs = fig_spec + fig_common
    save_html_report(
        save_dir=result_dir,
        save_name=f"emotion_face-{group_name}_group-analysis_report",
        figures=figs,
        title=f"面部情绪识别{group_name}组分析",
    )

    return {
        "all_results": all_results,
        "group_metrics": group_metrics,
        "statistical_results": statistical_results,
        "group_mean": group_mean_metrics,
        "group_std": group_std_metrics,
    }


def run_groups_emotion_analysis(
    control_files: list[Path],
    experimental_files: list[Path],
    result_dir: Path = Path("emotion_face_group_comparison_results"),
    groups: list[str] = None,
) -> dict[str, Any]:
    """比较对照组和实验组"""

    control_name = groups[0] if groups else "control"

    control_group_results = run_group_emotion_analysis(
        control_files, result_dir, control_name
    )
    control_results = control_group_results["all_results"]
    control_metrics = control_group_results["group_metrics"]

    experimental_name = groups[1] if groups and len(groups) > 1 else "experimental"

    experimental_group_results = run_group_emotion_analysis(
        experimental_files, result_dir, experimental_name
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

    if normality_results:
        normality_df = pd.DataFrame(normality_results).T
        normality_df.to_csv(result_dir / "normality_tests.csv")

    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results).T
        comparison_df.to_csv(result_dir / "group_comparisons.csv")

    fig_spec = create_group_comparison_visualizations(
        control_metrics, experimental_metrics
    )

    fig_common = create_common_comparison_figures(
        comparison_results, key_metrics, metric_names
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
        "control_metrics": control_metrics,
        "experimental_metrics": experimental_metrics,
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
