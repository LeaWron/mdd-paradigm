import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import statsmodels.api as sm
from omegaconf import DictConfig
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from psycho.analysis.utils import DataUtils, extract_trials_by_block

warnings.filterwarnings("ignore")


def load_and_prepare_data(
    df: pl.DataFrame,
    target_blocks=None,
    block_col="block_index",
    trial_col="trial_index",
    fill_na=True,
):
    raw_df = df
    print(f"原始数据形状: {raw_df.shape}")

    if target_blocks is None:
        target_blocks = [0, 1]

    processed_df = extract_trials_by_block(
        df=raw_df,
        target_block_indices=target_blocks,
        block_col=block_col,
        trial_col=trial_col,
        fill_na=fill_na,
    )

    print(f"处理后数据形状: {processed_df.shape}")

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

    processed_df = processed_df.with_columns(
        [
            pl.col("stim")
            .map_elements(map_stim_type, return_dtype=pl.Utf8, skip_nulls=False)
            .alias("stim_type"),
            pl.col("choice")
            .map_elements(map_stim_type, return_dtype=pl.Utf8, skip_nulls=False)
            .alias("choice_type"),
        ]
    )

    # [ ] 没有反应的是否要加惩罚处理
    # [ ] 反应太快的是否抛弃
    processed_df = processed_df.with_columns(
        pl.when(pl.col("rt") < 0.1)
        .then(0.1)
        .when(pl.col("rt") > 5.0)
        .then(5.0)
        .when(pl.col("rt").is_null())
        .then(pl.col("rt").mean() + 0.6)
        .otherwise(pl.col("rt"))
        .alias("rt_clean")
    )

    processed_df = processed_df.with_columns(
        (pl.col("stim_type") == pl.col("choice_type")).alias("correct")
    )

    return processed_df


def calculate_basic_metrics(df):
    metrics = {}

    # 总体指标
    metrics["overall_accuracy"] = df["correct"].mean()
    metrics["median_rt"] = df["rt_clean"].median()
    metrics["total_trials"] = len(df)

    # 分块正确率
    block_correct = (
        df.group_by("block_index")
        .agg(pl.col("correct").mean().alias("correct_rate"))
        .sort("block_index")
    )
    metrics["block_accuracy"] = block_correct

    # 按情绪类型统计
    emotion_correct = (
        df.group_by("stim_type")
        .agg(
            [
                pl.col("correct").mean().alias("correct_rate"),
                pl.col("correct").count().alias("trial_count"),
            ]
        )
        .sort("stim_type")
    )
    metrics["emotion_accuracy"] = emotion_correct

    # 反应时分析
    rt_summary = (
        df.group_by("stim_type")
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


def calculate_intensity_metrics(df):
    metrics = {}

    # 强度一致性分析（非中性刺激）
    df_intensity = df.filter(pl.col("stim_type") != "neutral").with_columns(
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
            # 转换为pandas进行相关性计算
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
    # 按刺激的实际类型，看被判断为中性的那些刺激的强度分布
    neutral_choices_by_stim = df.filter(pl.col("choice_type") == "neutral")

    if neutral_choices_by_stim.height > 0:
        # 统计不同刺激类型被判断为中性的情况
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

        # 统计被判断为中性的刺激中，各原始强度的分布
        intensity_distribution = (
            neutral_choices_by_stim.group_by(["stim_type", "label_intensity"])
            .agg(pl.count().alias("count"))
            .sort(["stim_type", "label_intensity"])
        )

        metrics["neutral_intensity_distribution"] = intensity_distribution

    # 按被试的选择，看那些被判断为中性的刺激的实际强度
    metrics["neutral_judgment_summary"] = (
        df.group_by("choice_type")
        .agg(
            [
                pl.count().alias("total_judgments"),
                pl.col("label_intensity").mean().alias("mean_label_intensity"),
                pl.col("label_intensity").std().alias("std_label_intensity"),
                pl.col("label_intensity").min().alias("min_label_intensity"),
                pl.col("label_intensity").max().alias("max_label_intensity"),
            ]
        )
        .sort("choice_type")
    )

    # 找出最可能被判断为中性的强度阈值
    # 对于每个刺激类型，计算被判断为中性的比例随强度的变化
    neutral_prob_by_intensity = []
    for stim_type in ["positive", "negative"]:
        stim_data = df.filter(pl.col("stim_type") == stim_type)
        if stim_data.height > 0:
            prob_data = (
                stim_data.group_by("label_intensity")
                .agg(
                    [
                        pl.count().alias("total"),
                        pl.col("choice_type")
                        .filter(pl.col("choice_type") == "neutral")
                        .count()
                        .alias("neutral_count"),
                    ]
                )
                .with_columns(
                    [
                        (pl.col("neutral_count") / pl.col("total")).alias(
                            "neutral_prob"
                        ),
                        pl.lit(stim_type).alias("stim_type"),
                    ]
                )
                .select(
                    [
                        "stim_type",
                        "label_intensity",
                        "total",
                        "neutral_count",
                        "neutral_prob",
                    ]
                )
            )
            neutral_prob_by_intensity.append(prob_data)

    if neutral_prob_by_intensity:
        metrics["neutral_probability_by_intensity"] = pl.concat(
            neutral_prob_by_intensity
        )

    return metrics


def perform_statistical_tests(df, df_intensity=None):
    results = {}

    # 转换为pandas以兼容统计库
    df_pd = df.to_pandas()

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


def create_visualizations(df: pl.DataFrame, metrics, result_dir):
    emotion_correct_pd = metrics["emotion_accuracy"].to_pandas()
    rt_summary_pd = metrics["reaction_time"].to_pandas()  # noqa: F841
    df_pd = df.to_pandas()  # noqa: F841

    # 强度数据
    df_intensity = df.filter(pl.col("stim_type") != "neutral")
    df_intensity_pd = df_intensity.to_pandas() if df_intensity.height > 0 else None

    # 创建主仪表板
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "不同情绪类型的识别正确率",
            "不同情绪类型的反应时分布",
            "反应时与正确率的关系",
            "强度评分一致性",
            "中性阈值分析",
            "分块正确率变化",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # 正确率分布
    fig.add_trace(
        go.Bar(
            x=emotion_correct_pd["stim_type"],
            y=emotion_correct_pd["correct_rate"],
            text=[f"{rate:.1%}" for rate in emotion_correct_pd["correct_rate"]],
            textposition="auto",
            marker_color=["#636efa", "#00cc96", "#ef553b"],
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(range=[0, 1.05], title_text="正确率", row=1, col=1)

    # 反应时分布
    for stim_type in ["positive", "neutral", "negative"]:
        rt_data = (
            df.filter(pl.col("stim_type") == stim_type)["rt_clean"]
            .drop_nulls()
            .to_list()
        )
        if rt_data:
            fig.add_trace(
                go.Box(
                    y=rt_data,
                    name=stim_type,
                    marker_color="lightseagreen",
                    boxmean="sd",
                ),
                row=1,
                col=2,
            )
    fig.update_yaxes(title_text="反应时 (秒)", row=1, col=2)

    # 正确率与反应时的关系
    scatter_sample = df.sample(fraction=0.3, seed=42).to_pandas()
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
                    title=dict(text="反应时与正确率情绪类型", side="top"),
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
        ),
        row=1,
        col=3,
    )
    fig.update_yaxes(
        tickvals=[0, 1], ticktext=["错误", "正确"], title_text="是否正确", row=1, col=3
    )
    fig.update_xaxes(title_text="反应时 (秒)", row=1, col=3)

    # 强度评分一致性
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
                        title=dict(text="强度评分一致性情绪类型", side="top"),
                        tickvals=[0, 1],
                        ticktext=["积极", "消极"],
                        len=0.1,
                        y=0.2,
                        x=1.1,
                        thickness=15,
                        orientation="h",
                    ),
                ),
                text=df_intensity_pd["stim_type"],
                hovertemplate="<b>标签强度</b>: %{x}<br><b>被试评分</b>: %{y}<br><b>类型</b>: %{text}<extra></extra>",
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
            ),
            row=2,
            col=1,
        )
    fig.update_xaxes(title_text="标签强度 (预设)", row=2, col=1)
    fig.update_yaxes(title_text="被试评分强度", row=2, col=1)

    # 中性阈值分析
    if "neutral_probability_by_intensity" in metrics:
        neutral_prob_data = metrics["neutral_probability_by_intensity"]
        for stim_type in ["positive", "negative"]:
            stim_prob = neutral_prob_data.filter(pl.col("stim_type") == stim_type)
            if stim_prob.height > 0:
                stim_prob_pd = stim_prob.to_pandas()
                fig.add_trace(
                    go.Scatter(
                        x=stim_prob_pd["label_intensity"],
                        y=stim_prob_pd["neutral_prob"],
                        mode="markers",
                        name=f"{stim_type}被判断为中性的概率",
                        line=dict(width=2),
                        marker=dict(size=8),
                    ),
                    row=2,
                    col=2,
                )
    fig.update_yaxes(range=[0, 1.05], title_text="被判断为中性的概率", row=2, col=2)
    fig.update_xaxes(title_text="刺激标签强度", row=2, col=2)

    # 分块正确率变化
    block_correct_pd = metrics["block_accuracy"].to_pandas()
    fig.add_trace(
        go.Scatter(
            x=block_correct_pd["block_index"],
            y=block_correct_pd["correct_rate"],
            mode="lines+markers",
            marker=dict(size=12),
            line=dict(width=3),
            hovertemplate="<b>区块</b>: %{x}<br><b>正确率</b>: %{y:.1%}<extra></extra>",
        ),
        row=2,
        col=3,
    )
    fig.update_yaxes(range=[0, 1.05], title_text="正确率", row=2, col=3)
    fig.update_xaxes(title_text="区块编号", row=2, col=3)

    # 更新整体布局
    fig.update_layout(
        height=900,
        width=1400,
        title_text="面部情绪识别实验行为数据分析",
        title_font_size=20,
        showlegend=True,
        hovermode="closest",
    )

    # 保存图表
    html_path = result_dir / "facial_emotion_analysis.html"
    png_path = result_dir / "facial_emotion_analysis.png"

    fig.write_html(str(html_path))
    fig.write_image(str(png_path), scale=2)

    # 创建中性阈值专用图表
    if "neutral_threshold_by_stimulus" in metrics:
        create_neutral_threshold_visualization(metrics, result_dir)

    return {"main_dashboard_html": html_path, "main_dashboard_png": png_path}


def create_neutral_threshold_visualization(metrics, result_dir):
    """创建中性阈值专用可视化"""
    if "neutral_threshold_by_stimulus" not in metrics:
        return

    threshold_data = metrics["neutral_threshold_by_stimulus"].to_pandas()

    fig = go.Figure()

    # 柱状图：不同刺激类型被判断为中性的次数
    fig.add_trace(
        go.Bar(
            x=threshold_data["stim_type"],
            y=threshold_data["count"],
            name="被判断为中性的次数",
            text=threshold_data["count"],
            textposition="auto",
        )
    )

    # 折线图：被判断为中性的刺激的平均强度
    fig.add_trace(
        go.Scatter(
            x=threshold_data["stim_type"],
            y=threshold_data["mean_label_intensity"],
            mode="lines+markers",
            name="平均标签强度",
            yaxis="y2",
            line=dict(color="red", width=2),
            marker=dict(size=10),
        )
    )

    fig.update_layout(
        title="中性阈值分析：不同刺激类型被判断为中性情况",
        yaxis=dict(title="被判断为中性的次数"),
        yaxis2=dict(title="平均标签强度", overlaying="y", side="right", range=[0, 10]),
        hovermode="x unified",
    )

    threshold_path = result_dir / "neutral_threshold_analysis.html"
    fig.write_html(str(threshold_path))

    return threshold_path


def save_results(df, metrics, stats_results, viz_paths, result_dir):
    def convert_to_serializable(obj):
        """递归转换对象为JSON可序列化格式"""
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            # 将DataFrame转换为字典列表
            return obj.to_dict(orient="records")
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            # 转换numpy数值类型为Python内置类型
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            # 其他类型（字符串、整数、浮点数、布尔值、None）直接返回
            return obj

    # 转换stats_results
    serializable_stats = convert_to_serializable(stats_results)

    saved_files = {}

    # 处理后的数据
    data_path = result_dir / "processed_data.csv"
    df.write_csv(str(data_path))
    saved_files["processed_data"] = data_path

    # 指标结果
    metrics_path = result_dir / "metrics_summary.json"

    # JSON序列化
    json_metrics = {
        "overall_accuracy": float(metrics["overall_accuracy"]),
        "median_rt": float(metrics["median_rt"]),
        "total_trials": int(metrics["total_trials"]),
    }

    if "intensity_correlation" in metrics:
        json_metrics["intensity_correlation"] = metrics["intensity_correlation"]

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(json_metrics, f, ensure_ascii=False, indent=2)
    saved_files["metrics_summary"] = metrics_path

    # 详细统计表
    for metric_name, metric_df in metrics.items():
        if isinstance(metric_df, pl.DataFrame):
            csv_path = result_dir / f"{metric_name}.csv"
            metric_df.write_csv(str(csv_path))
            saved_files[metric_name] = csv_path

    # 统计检验结果
    stats_path = result_dir / "statistical_tests.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
    saved_files["statistical_tests"] = stats_path

    # 可视化文件路径
    saved_files.update(viz_paths)

    return saved_files


def analyze_emotion_data(
    df, target_blocks=None, block_col="block_index", result_dir=None
):
    # 预处理
    processed_df = load_and_prepare_data(
        df=df,
        target_blocks=target_blocks,
        block_col=block_col,
        fill_na=True,
    )

    print("计算基本行为指标...")
    metrics = calculate_basic_metrics(processed_df)

    print("计算强度指标和中性阈值...")
    intensity_metrics = calculate_intensity_metrics(processed_df)
    metrics.update(intensity_metrics)

    print("执行统计检验...")
    stats_results = perform_statistical_tests(processed_df)

    print("创建可视化图表...")
    viz_paths = create_visualizations(processed_df, metrics, result_dir)

    print("保存分析结果...")
    saved_files = save_results(
        processed_df, metrics, stats_results, viz_paths, result_dir
    )

    print("\n" + "=" * 60)
    print("摘要")
    print("=" * 60)
    print(f"总体正确率: {metrics['overall_accuracy']:.2%}")
    print(f"中位反应时: {metrics['median_rt']:.3f} 秒")

    if "neutral_threshold_by_stimulus" in metrics:
        print("\n中性阈值分析:")
        threshold_data = metrics["neutral_threshold_by_stimulus"]
        for row in threshold_data.iter_rows():
            stim_type, count, mean_intensity = row[0], row[1], row[2]
            if count > 0:
                print(
                    f"  {stim_type}刺激被判断为中性: {count}次, 平均强度: {mean_intensity:.2f}"
                )

    if "intensity_correlation" in metrics:
        corr_info = metrics["intensity_correlation"]
        print(f"\n强度评分相关性: r = {corr_info['r']:.3f}, p = {corr_info['p']:.4f}")

    return {
        "processed_data": processed_df,
        "metrics": metrics,
        "statistical_results": stats_results,
        "visualization_paths": viz_paths,
        "saved_files": saved_files,
    }


def run_emotion_analysis(cfg: DictConfig = None, data_utils: DataUtils = None):
    print("=" * 60)
    print("面部情绪识别分析")
    print("=" * 60)

    if data_utils is None:
        file_input = input("请输入数据文件路径: \n").strip("'").strip()
        file_path = Path(file_input.strip("'").strip('"')).resolve()
    else:
        file_path = (
            Path(cfg.output_dir)
            / data_utils.date
            / f"{data_utils.session_id}-face_recognition.csv"
        )

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    if cfg is None:
        result_dir = file_path.parent.parent / "results"
    else:
        result_dir = Path(cfg.result_dir)
    if data_utils is not None:
        result_dir = result_dir / str(data_utils.session_id)
    result_dir = result_dir / "emotion_analysis"

    result_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(file_path)

    results = analyze_emotion_data(
        df=df,
        target_blocks=[0, 1],
        block_col="block_index",
        result_dir=result_dir,
    )

    print("\n" + "=" * 60)
    print("分析完成！")
    print(f"结果已保存至: {result_dir}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_emotion_analysis()
