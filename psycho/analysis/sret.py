import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl
from omegaconf import DictConfig
from scipy import stats

from psycho.analysis.utils import DataUtils, extract_trials_by_block

PICTURE_REFERENCE = {
    # 注意：这些值是估计值，需要根据实际情况调整
    "endorsement_count": {
        "HCO": {"positive": 30, "negative": 2},
        "MDD": {"positive": 12, "negative": 24},
    },
    "reaction_time": {
        "HCO": {"positive": 500, "negative": 1150},
        "MDD": {"positive": 900, "negative": 600},
    },
}


def preprocess_sret_data(
    df: pl.DataFrame,
    target_blocks: list[int] = None,
    block_col: str = "block_index",
) -> pl.DataFrame:
    print("1. 数据预处理...")

    if target_blocks is not None:
        print(f"  提取区块 {target_blocks} 的数据...")
        df_processed = extract_trials_by_block(
            df,
            target_block_indices=target_blocks,
            block_col=block_col,
            fill_na=True,
        )
        print(f"  提取后数据形状: {df_processed.shape}")
    else:
        df_processed = df.clone()
        print(f"  使用所有数据，形状: {df_processed.shape}")

    # 检查必要列
    required_columns = ["stim_word", "response", "rt"]
    missing_columns = [
        col for col in required_columns if col not in df_processed.columns
    ]
    if missing_columns:
        print(f"  ❌ 缺少必要列: {missing_columns}")
        return None

    # 转换反应时为毫秒
    df_processed = df_processed.with_columns(
        (
            pl.when(pl.col("rt").is_null())
            .then(pl.col("rt").max() + 1)
            .otherwise(pl.col("rt"))
        ),
    )
    df_processed = df_processed.with_columns(
        (pl.col("rt") * 1000).alias("rt_ms"),
    )

    # 处理强度评分（如果有）
    if "intensity" in df_processed.columns:
        # 确保强度评分在0-10范围内
        df_processed = df_processed.with_columns(
            pl.col("intensity").clip(0, 10).alias("intensity")
        )

    # 添加反应类型编码
    df_processed = df_processed.with_columns(
        (pl.col("response") == "yes").cast(pl.Int8).alias("response_code")
    )

    return df_processed


def analyze_basic(df: pl.DataFrame) -> dict:
    print("\n2. 基础分析...")

    results = {}

    # 总试次和认同率
    total_trials = df.height
    yes_count = df.filter(pl.col("response") == "yes").height
    no_count = df.filter(pl.col("response") == "no").height
    endorsement_rate = yes_count / total_trials if total_trials > 0 else 0

    results["total_trials"] = total_trials
    results["yes_count"] = yes_count
    results["no_count"] = no_count
    results["endorsement_rate"] = endorsement_rate

    print(f"  总试次数: {total_trials}")
    print(f"  认同数量: {yes_count}")
    print(f"  不认同数量: {no_count}")
    print(f"  认同率: {endorsement_rate:.2%}")

    # 反应时统计
    rt_stats = df.select(
        [
            pl.col("rt_ms").mean().alias("mean_rt"),
            pl.col("rt_ms").std().alias("std_rt"),
            pl.col("rt_ms").median().alias("median_rt"),
            pl.col("rt_ms").min().alias("min_rt"),
            pl.col("rt_ms").max().alias("max_rt"),
        ]
    )

    results["rt_stats"] = rt_stats.to_dicts()[0]

    print(f"  平均反应时: {rt_stats['mean_rt'][0]:.1f} ms")
    print(f"  反应时标准差: {rt_stats['std_rt'][0]:.1f} ms")
    print(f"  反应时中位数: {rt_stats['median_rt'][0]:.1f} ms")

    # 反应类型分析
    response_stats = (
        df.group_by("response")
        .agg(
            [
                pl.col("rt_ms").mean().alias("mean_rt"),
                pl.col("rt_ms").std().alias("std_rt"),
                pl.count().alias("count"),
            ]
        )
        .sort("response")
    )

    if "intensity" in df.columns:
        intensity_stats = (
            df.group_by("response")
            .agg(
                [
                    pl.col("intensity").mean().alias("mean_intensity"),
                    pl.col("intensity").std().alias("std_intensity"),
                ]
            )
            .sort("response")
        )
        results["intensity_stats"] = intensity_stats.to_dicts()

    results["response_stats"] = response_stats.to_dicts()

    # 计算反应时差异
    if len(response_stats) >= 2:
        yes_rt = response_stats.filter(pl.col("response") == "yes")["mean_rt"][0]
        no_rt = response_stats.filter(pl.col("response") == "no")["mean_rt"][0]
        results["rt_difference_yes_no"] = yes_rt - no_rt
        print(f"  认同-不认同RT差异: {results['rt_difference_yes_no']:.1f} ms")

    return results


def analyze_valence(df: pl.DataFrame) -> dict:
    print("\n3. 词性分析...")

    results = {}

    # 按词性分组统计
    valence_stats = df.group_by("stim_type").agg(
        [
            pl.col("response")
            .filter(pl.col("response") == "yes")
            .count()
            .alias("yes_count"),
            pl.col("response").count().alias("total_count"),
            pl.col("rt_ms").mean().alias("mean_rt"),
            pl.col("rt_ms").std().alias("std_rt"),
        ]
    )

    # 如果有强度评分
    if "intensity" in df.columns:
        intensity_by_valence = df.group_by("stim_type").agg(
            [
                pl.col("intensity").mean().alias("mean_intensity"),
                pl.col("intensity").std().alias("std_intensity"),
            ]
        )
        valence_stats = valence_stats.join(intensity_by_valence, on="stim_type")

    # 计算认同率
    valence_stats = valence_stats.with_columns(
        [(pl.col("yes_count") / pl.col("total_count")).alias("endorsement_rate")]
    )

    results["valence_stats"] = valence_stats.to_dicts()

    # 转换为字典便于访问
    valence_dict = {}
    for row in valence_stats.to_dicts():
        valence = row["stim_type"]
        valence_dict[valence] = row

    # 打印结果
    for valence in ["positive", "negative"]:
        if valence in valence_dict:
            data = valence_dict[valence]
            print(f"  {valence}词:")
            print(f"    认同数量: {data['yes_count']}/{data['total_count']}")
            print(f"    认同率: {data['endorsement_rate']:.2%}")
            print(f"    平均反应时: {data['mean_rt']:.1f} ms")
            if "mean_intensity" in data:
                print(f"    平均强度: {data['mean_intensity']:.2f}")

    return results


def calculate_key_metrics(df: pl.DataFrame, valence_results: dict) -> dict:
    print("\n4. 计算关键指标...")

    metrics = {}

    # 基础指标
    total = df.height
    yes_count = df.filter(pl.col("response") == "yes").height
    no_count = df.filter(pl.col("response") == "no").height  # noqa: F841

    # 1. 认同率: 认同数量 / 总数量
    endorsement_rate = yes_count / total if total > 0 else 0
    metrics["endorsement_rate"] = endorsement_rate

    # 2. 积极偏向: 积极认同率 - 消极认同率
    if "valence_stats" in valence_results:
        valence_data = {}
        for stat in valence_results["valence_stats"]:
            valence_data[stat["stim_type"]] = stat

        if "positive" in valence_data and "negative" in valence_data:
            positive_rate = valence_data["positive"]["endorsement_rate"]
            negative_rate = valence_data["negative"]["endorsement_rate"]
            metrics["positive_bias"] = positive_rate - negative_rate

    # 3. 消极RT - 积极RT
    if "valence_stats" in valence_results:
        valence_data = {}
        for stat in valence_results["valence_stats"]:
            valence_data[stat["stim_type"]] = stat

        if "positive" in valence_data and "negative" in valence_data:
            negative_rt = valence_data["negative"]["mean_rt"]
            positive_rt = valence_data["positive"]["mean_rt"]
            metrics["rt_negative_minus_positive"] = negative_rt - positive_rt

    # 4. 认同RT - 不认同RT
    yes_rt = df.filter(pl.col("response") == "yes")["rt_ms"].mean()
    no_rt = df.filter(pl.col("response") == "no")["rt_ms"].mean()
    metrics["rt_endorsed_minus_not"] = yes_rt - no_rt

    # 5. 消极intensity - 积极intensity
    if "intensity" in df.columns and "valence_stats" in valence_results:
        valence_data = {}
        for stat in valence_results["valence_stats"]:
            valence_data[stat["stim_type"]] = stat

        if "positive" in valence_data and "negative" in valence_data:
            if (
                "mean_intensity" in valence_data["negative"]
                and "mean_intensity" in valence_data["positive"]
            ):
                negative_intensity = valence_data["negative"]["mean_intensity"]
                positive_intensity = valence_data["positive"]["mean_intensity"]
                metrics["intensity_negative_minus_positive"] = (
                    negative_intensity - positive_intensity
                )

    # 打印所有指标
    print("\n  关键指标计算结果:")
    print(f"  1. 认同率: {metrics.get('endorsement_rate', 0):.2%}")
    if "positive_bias" in metrics:
        print(f"  2. 积极偏向: {metrics['positive_bias']:.3f} (正值表示积极偏向)")
    if "rt_negative_minus_positive" in metrics:
        print(f"  3. 消极RT - 积极RT: {metrics['rt_negative_minus_positive']:.1f} ms")
    if "rt_endorsed_minus_not" in metrics:
        print(f"  4. 认同RT - 不认同RT: {metrics['rt_endorsed_minus_not']:.1f} ms")
    if "intensity_negative_minus_positive" in metrics:
        print(
            f"  5. 消极intensity - 积极intensity: {metrics['intensity_negative_minus_positive']:.2f}"
        )

    return metrics


def create_valence_analysis_plot(valence_results: dict, result_dir: Path):
    if "valence_stats" not in valence_results:
        return

    # 准备数据
    valence_data = valence_results["valence_stats"]

    # 按词性组织数据
    valence_dict = {}
    for row in valence_data:
        valence = row["stim_type"]
        valence_dict[valence] = row

    # 创建子图
    fig = sp.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("认同数量", "反应时间"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.2,
    )

    # 图表1：认同数量
    if "positive" in valence_dict and "negative" in valence_dict:
        pos_data = valence_dict["positive"]
        neg_data = valence_dict["negative"]

        fig.add_trace(
            go.Bar(
                name="积极词",
                x=["积极词"],
                y=[pos_data["yes_count"]],
                marker_color="lightblue",
                text=[f"{pos_data['yes_count']}"],
                textposition="auto",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                name="消极词",
                x=["消极词"],
                y=[neg_data["yes_count"]],
                marker_color="lightcoral",
                text=[f"{neg_data['yes_count']}"],
                textposition="auto",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=["积极词", "消极词"],
                y=[
                    PICTURE_REFERENCE["endorsement_count"]["HCO"]["positive"],
                    PICTURE_REFERENCE["endorsement_count"]["HCO"]["negative"],
                ],
                mode="markers",
                name="对照组参考",
                marker=dict(size=12, color="red", symbol="diamond"),
                opacity=0.7,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=["积极词", "消极词"],
                y=[
                    PICTURE_REFERENCE["endorsement_count"]["MDD"]["positive"],
                    PICTURE_REFERENCE["endorsement_count"]["MDD"]["negative"],
                ],
                mode="markers",
                name="MDD参考",
                marker=dict(size=12, color="green", symbol="diamond"),
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

    # 图表2：反应时间
    if "positive" in valence_dict and "negative" in valence_dict:
        pos_data = valence_dict["positive"]
        neg_data = valence_dict["negative"]

        fig.add_trace(
            go.Bar(
                name="积极词",
                x=["积极词"],
                y=[pos_data["mean_rt"]],
                marker_color="lightblue",
                text=[f"{pos_data['mean_rt']:.0f} ms"],
                textposition="auto",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Bar(
                name="消极词",
                x=["消极词"],
                y=[neg_data["mean_rt"]],
                marker_color="lightcoral",
                text=[f"{neg_data['mean_rt']:.0f} ms"],
                textposition="auto",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=["积极词", "消极词"],
                y=[
                    PICTURE_REFERENCE["reaction_time"]["HCO"]["positive"],
                    PICTURE_REFERENCE["reaction_time"]["HCO"]["negative"],
                ],
                mode="markers",
                name="对照组参考",
                marker=dict(size=12, color="red", symbol="diamond"),
                opacity=0.7,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=["积极词", "消极词"],
                y=[
                    PICTURE_REFERENCE["reaction_time"]["MDD"]["positive"],
                    PICTURE_REFERENCE["reaction_time"]["MDD"]["negative"],
                ],
                mode="markers",
                name="MDD参考",
                marker=dict(size=12, color="green", symbol="diamond"),
                opacity=0.7,
            ),
            row=1,
            col=2,
        )

    # 更新布局
    fig.update_layout(
        title="词性分析",
        template="plotly_white",
        showlegend=True,
        height=400,
        width=800,
    )

    fig.update_yaxes(title_text="认同数量", row=1, col=1)
    fig.update_yaxes(title_text="反应时间 (ms)", row=1, col=2)

    fig.write_html(result_dir / "key_analysis.html")
    # fig.show()


def create_rt_distribution_plot(df_pd: pd.DataFrame, result_dir: Path):
    fig = px.histogram(
        df_pd,
        x="rt_ms",
        color="response",
        nbins=30,
        title="反应时分布",
        labels={"rt_ms": "反应时 (ms)", "count": "频数"},
        opacity=0.7,
        barmode="overlay",
    )
    fig.update_layout(
        xaxis_range=[0, df_pd["rt_ms"].max() * 1.1],
        template="plotly_white",
    )
    fig.write_html(result_dir / "rt_distribution.html")
    # fig.show()


def create_key_metrics_plot(key_metrics: dict, result_dir: Path):
    # 要显示的指标
    display_metrics = {
        "positive_bias": "积极偏向",
        "rt_negative_minus_positive": "消极-积极RT差",
        "rt_endorsed_minus_not": "认同-不认同RT差",
        "endorsement_rate": "认同率",
    }

    # 准备数据
    metric_names = []
    metric_values = []
    metric_colors = []

    for metric_key, metric_name in display_metrics.items():
        if metric_key in key_metrics:
            metric_names.append(metric_name)
            value = key_metrics[metric_key]
            metric_values.append(abs(value) if metric_key != "positive_bias" else value)

            if metric_key == "positive_bias":
                if value > 0.1:
                    metric_colors.append("lightblue")  # 强积极偏向
                elif value > 0:
                    metric_colors.append("lightgreen")  # 弱积极偏向
                elif value < -0.1:
                    metric_colors.append("lightcoral")  # 强消极偏向
                else:
                    metric_colors.append("lightgray")  # 中性
            else:
                metric_colors.append("lightskyblue")

    if not metric_names:
        return

    # 创建条形图
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles="关键指标")
    fig.add_trace(
        go.Bar(
            x=["消极-积极RT差", "认同-不认同RT差"],
            y=[
                key_metrics["rt_negative_minus_positive"],
                key_metrics["rt_endorsed_minus_not"],
            ],
            marker_color=["lightcoral", "lightskyblue"],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=["积极偏向", "认同率"],
            y=[key_metrics["positive_bias"], key_metrics["endorsement_rate"]],
            marker_color=["lightblue", "lightgreen"],
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="关键指标可视化",
        xaxis_title="指标",
        yaxis_title="数值",
        template="plotly_white",
        showlegend=True,
    )

    # 添加参考线（针对积极偏向）
    if "积极偏向" in metric_names:
        idx = metric_names.index("积极偏向")  # noqa: F841
        fig.add_hline(
            y=0.1,
            line_dash="dash",
            line_color="blue",
            annotation_text="HC阈值",
            annotation_position="top right",
            row=1,
            col=2,
        )
        fig.add_hline(
            y=-0.1,
            line_dash="dash",
            line_color="red",
            annotation_text="MDD阈值",
            annotation_position="bottom right",
            row=1,
            col=2,
        )

    fig.write_html(result_dir / "key_metrics.html")
    # fig.show()


def create_rt_intensity_plot(df_pd, result_dir: Path):
    """创建反应时与强度关系图"""
    # [ ] 未必是线性拟合, 先用着
    fig = px.scatter(
        df_pd,
        x="rt_ms",
        y="intensity",
        color="response",
        title="反应时与强度关系",
        labels={"rt_ms": "反应时 (ms)", "intensity": "强度评分 (0-10)"},
        hover_data=["stim_word"],
        trendline="lowess",
    )
    # ['lowess', 'rolling', 'ewm', 'expanding', 'ols']
    fig.update_layout(template="plotly_white")
    fig.write_html(result_dir / "rt_intensity.html")
    # fig.show()


def create_visualizations(
    df: pl.DataFrame,
    basic_results: dict,
    valence_results: dict,
    key_metrics: dict,
    result_dir: Path,
):
    print("\n6. 生成可视化图表...")

    # 转换为pandas用于Plotly
    df_pd = df.to_pandas() if df.height < 10000 else df.head(1000).to_pandas()

    # 图表1：词性分析图
    if "valence_stats" in valence_results:
        create_valence_analysis_plot(valence_results, result_dir)

    # 图表2：反应时分布
    create_rt_distribution_plot(df_pd, result_dir)

    # 图表3：关键指标可视化
    create_key_metrics_plot(key_metrics, result_dir)

    # 图表4：反应时与强度关系（如果有强度评分）
    if "intensity" in df.columns:
        create_rt_intensity_plot(df_pd, result_dir)


def save_results(results: dict, result_dir: Path):
    print("\n7. 保存分析结果...")

    with open(result_dir / "sret_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 关键指标
    if "key_metrics" in results:
        metrics_data = []
        for key, value in results["key_metrics"].items():
            metrics_data.append({"指标": key, "值": value})

        metrics_df = pl.DataFrame(metrics_data)
        metrics_df.write_csv(result_dir / "key_metrics.csv")

        print("  关键指标:")
        for row in metrics_data:
            print(f"    {row['指标']}: {row['值']:.4f}")

    print(f"  结果已保存到: {result_dir}")


def analyze_sret_data(
    df: pl.DataFrame,
    target_blocks: int | list[int] = None,
    block_col: str = "block_index",
    result_dir: Path = None,
) -> dict:
    # 数据预处理
    df_processed = preprocess_sret_data(df, target_blocks, block_col)
    if df_processed is None:
        return {}

    # 基础分析
    basic_results = analyze_basic(df_processed)

    # 词性分析
    if "stim_type" in df_processed.columns:
        valence_results = analyze_valence(df_processed)
    else:
        valence_results = {}
        warnings.warn("未找到stim_type列，跳过词性分析")

    # 计算关键指标
    key_metrics = calculate_key_metrics(df_processed, valence_results)

    # 生成可视化
    create_visualizations(
        df_processed, basic_results, valence_results, key_metrics, result_dir
    )

    # 保存结果
    results = {
        "basic": basic_results,
        "valence": valence_results,
        "key_metrics": key_metrics,
    }

    save_results(results, result_dir)

    return results


def run_sret_analysis(cfg: DictConfig = None, data_utils: DataUtils = None):
    print("=" * 60)
    print("自我参照编码任务分析")
    print("=" * 60)
    if data_utils is None:
        file_input = input("请输入数据文件路径: \n").strip("'").strip()
        file_path = Path(file_input.strip("'").strip('"')).resolve()
    else:
        file_path = (
            Path(cfg.output_dir) / data_utils.date / f"{data_utils.session_id}-sret.csv"
        )

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    df = pl.read_csv(file_path)

    if cfg is None:
        result_dir = file_path.parent.parent / "results"
    else:
        result_dir = Path(cfg.result_dir)
    if data_utils is not None:
        result_dir = result_dir / str(data_utils.session_id)
    result_dir = result_dir / "sret_analysis"

    result_dir.mkdir(parents=True, exist_ok=True)
    analyze_sret_data(
        df=df,
        target_blocks=["Encoding"],  # 目标区块
        block_col="phase",  # 区块列名
        result_dir=result_dir,
    )

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_sret_analysis()
