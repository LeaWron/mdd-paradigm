import json
import warnings
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl
from scipy import stats

from psycho.analysis.utils import extract_trials_by_block

# ============================================================================
# 图片参考值（基于图片估计）
# ============================================================================

PICTURE_REFERENCE = {
    # 注意：这些值是估计值，需要根据实际情况调整
    "endorsement_count": {
        "HCO": {"positive": 1400, "negative": 400},  # 估计值
        "MDD": {"positive": 600, "negative": 1000},  # 估计值
    },
    "reaction_time": {
        "HCO": {"positive": 1000, "negative": 1200},  # 估计值
        "MDD": {"positive": 1400, "negative": 1600},  # 估计值
    },
    # 关键指标参考模式（从图片推断）
    "key_indicators": {
        "positive_bias_hco": 0.3,  # HCO组的积极偏向
        "positive_bias_mdd": -0.2,  # MDD组的消极偏向
        "rt_neg_pos_difference": 200,  # 消极-积极词RT差异
        "endorsed_not_difference": 100,  # 认同-不认同RT差异
    },
}


# ============================================================================
# 辅助函数
# ============================================================================


def _interpret_positive_bias(bias: float) -> str:
    """解释积极偏向"""
    if bias > 0.1:
        return "强烈的积极自我概念偏向（类似HCO组模式）"
    elif bias > 0:
        return "轻微的积极自我概念偏向"
    elif bias < -0.1:
        return "强烈的消极自我概念偏向（类似MDD组模式）"
    elif bias < 0:
        return "轻微的消极自我概念偏向"
    else:
        return "中性，无偏向"


def _interpret_rt_difference(diff: float) -> str:
    """解释RT差异"""
    if diff > 300:
        return "处理消极词明显慢于积极词"
    elif diff > 100:
        return "处理消极词慢于积极词（正常范围）"
    elif diff > -100:
        return "处理积极词和消极词速度相当"
    else:
        return "处理积极词慢于消极词（非典型）"


def _interpret_endorsed_rt_difference(diff: float) -> str:
    """解释认同-不认同RT差异"""
    if diff > 200:
        return "认同判断明显慢于不认同判断"
    elif diff > 50:
        return "认同判断慢于不认同判断（典型）"
    elif diff > -50:
        return "认同与不认同判断速度相当"
    else:
        return "认同判断快于不认同判断（非典型）"


# ============================================================================
# 数据预处理函数
# ============================================================================


def preprocess_sret_data(
    df: pl.DataFrame,
    target_blocks: list[int] = None,
    block_col: str = "block_index",
) -> pl.DataFrame:
    print("1. 数据预处理...")

    # 提取目标区块
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
    df_processed = df_processed.with_columns((pl.col("rt") * 1000).alias("rt_ms"))

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


# ============================================================================
# 分析函数
# ============================================================================


def analyze_basic(df: pl.DataFrame) -> dict:
    """基础分析"""
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
    """词性分析（积极/消极）"""
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
    """计算关键指标"""
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


def compare_with_picture_reference(key_metrics: dict) -> dict:
    """与图片参考值比较"""
    print("\n5. 与图片参考值比较...")

    comparison = {}
    pic_ref = PICTURE_REFERENCE["key_indicators"]

    print("  图片参考模式:")
    print(
        f"  1. 积极偏向参考值: HCO组≈{pic_ref['positive_bias_hco']:.2f}, MDD组≈{pic_ref['positive_bias_mdd']:.2f}"
    )
    print(f"  2. 消极-积极词RT差异: ≈{pic_ref['rt_neg_pos_difference']} ms")
    print(f"  3. 认同-不认同RT差异: ≈{pic_ref['endorsed_not_difference']} ms")

    # 比较每个指标
    for metric_name, metric_value in key_metrics.items():
        if metric_name == "positive_bias":
            # 比较积极偏向
            comparison[metric_name] = {
                "value": metric_value,
                "interpretation": _interpret_positive_bias(metric_value),
            }
            print("\n  积极偏向分析:")
            print(f"    你的数据: {metric_value:.3f}")
            print(f"    解释: {comparison[metric_name]['interpretation']}")

        elif metric_name == "rt_negative_minus_positive":
            # 比较RT差异
            ref_value = pic_ref["rt_neg_pos_difference"]
            difference = metric_value - ref_value
            percent_diff = (difference / ref_value) * 100 if ref_value != 0 else 0

            comparison[metric_name] = {
                "value": metric_value,
                "reference": ref_value,
                "difference": difference,
                "percent_difference": percent_diff,
                "interpretation": _interpret_rt_difference(metric_value),
            }

            print("\n  消极-积极词RT差异分析:")
            print(f"    你的数据: {metric_value:.1f} ms")
            print(f"    图片参考: {ref_value} ms")
            print(f"    差异: {difference:.1f} ms ({percent_diff:.1f}%)")
            print(f"    解释: {comparison[metric_name]['interpretation']}")

        elif metric_name == "rt_endorsed_minus_not":
            # 比较认同-不认同RT差异
            ref_value = pic_ref["endorsed_not_difference"]
            difference = metric_value - ref_value
            percent_diff = (difference / ref_value) * 100 if ref_value != 0 else 0

            comparison[metric_name] = {
                "value": metric_value,
                "reference": ref_value,
                "difference": difference,
                "percent_difference": percent_diff,
                "interpretation": _interpret_endorsed_rt_difference(metric_value),
            }

            print("\n  认同-不认同RT差异分析:")
            print(f"    你的数据: {metric_value:.1f} ms")
            print(f"    图片参考: {ref_value} ms")
            print(f"    差异: {difference:.1f} ms ({percent_diff:.1f}%)")
            print(f"    解释: {comparison[metric_name]['interpretation']}")

    return comparison


# ============================================================================
# 可视化函数
# ============================================================================


def create_valence_analysis_plot(valence_results: dict, result_dir: Path):
    """创建词性分析图（类似参考图片）"""
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

    # 更新布局
    fig.update_layout(
        title="词性分析（类似参考图片）",
        template="plotly_white",
        showlegend=True,
        height=400,
        width=800,
    )

    fig.update_yaxes(title_text="认同数量", row=1, col=1)
    fig.update_yaxes(title_text="反应时间 (ms)", row=1, col=2)

    fig.write_html(result_dir / "valence_analysis.html")
    fig.show()


def create_rt_distribution_plot(df_pd, result_dir: Path):
    """创建反应时分布图"""
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
    fig.show()


def create_key_metrics_plot(key_metrics: dict, result_dir: Path):
    """创建关键指标可视化"""
    # 筛选要显示的指标
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

            # 根据值设置颜色
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
    fig = go.Figure(
        data=[
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=metric_colors,
                text=[
                    f"{v:.3f}" if metric_names[i] == "积极偏向" else f"{v:.1f}"
                    for i, v in enumerate(metric_values)
                ],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="关键指标可视化",
        xaxis_title="指标",
        yaxis_title="数值",
        template="plotly_white",
        showlegend=False,
    )

    # 添加参考线（针对积极偏向）
    if "积极偏向" in metric_names:
        idx = metric_names.index("积极偏向")  # noqa: F841
        fig.add_hline(
            y=0.1,
            line_dash="dash",
            line_color="blue",
            annotation_text="HCO模式阈值",
            annotation_position="top right",
        )
        fig.add_hline(
            y=-0.1,
            line_dash="dash",
            line_color="red",
            annotation_text="MDD模式阈值",
            annotation_position="bottom right",
        )

    fig.write_html(result_dir / "key_metrics.html")
    fig.show()


def create_rt_intensity_plot(df_pd, result_dir: Path):
    """创建反应时与强度关系图"""
    fig = px.scatter(
        df_pd,
        x="rt_ms",
        y="intensity",
        color="response",
        title="反应时与强度关系",
        labels={"rt_ms": "反应时 (ms)", "intensity": "强度评分 (0-10)"},
        hover_data=["stim_word"],
        trendline="ols",
    )
    fig.update_layout(template="plotly_white")
    fig.write_html(result_dir / "rt_intensity.html")
    fig.show()


def create_visualizations(
    df: pl.DataFrame,
    basic_results: dict,
    valence_results: dict,
    key_metrics: dict,
    result_dir: Path,
):
    """创建可视化图表"""
    print("\n6. 生成可视化图表...")

    # 转换为pandas用于Plotly
    df_pd = df.to_pandas() if df.height < 10000 else df.head(1000).to_pandas()

    # 图表1：词性分析图（类似参考图片）
    if "valence_stats" in valence_results:
        create_valence_analysis_plot(valence_results, result_dir)

    # 图表2：反应时分布
    create_rt_distribution_plot(df_pd, result_dir)

    # 图表3：关键指标可视化
    create_key_metrics_plot(key_metrics, result_dir)

    # 图表4：反应时与强度关系（如果有强度评分）
    if "intensity" in df.columns:
        create_rt_intensity_plot(df_pd, result_dir)


# ============================================================================
# 保存结果函数
# ============================================================================


def save_results(results: dict, result_dir: Path):
    """保存分析结果"""
    print("\n7. 保存分析结果...")

    # 保存为JSON
    with open(result_dir / "sret_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存关键指标为CSV
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


# ============================================================================
# 主分析函数
# ============================================================================


def analyze_sret_data(
    df: pl.DataFrame,
    target_blocks: int | list[int] = None,
    block_col: str = "block_index",
    result_dir: Path = None,
) -> dict:
    """
    主分析函数

    Parameters:
    -----------
    df : pl.DataFrame
        原始数据框
    target_blocks : Optional[Union[int, List[int]]]
        目标区块索引
    block_col : str
        区块列名
    result_dir : Optional[Path]
        结果保存目录

    Returns:
    --------
    Dict: 分析结果
    """
    print("=" * 60)
    print("自我参照编码任务分析")
    print("=" * 60)

    # 设置结果目录
    if result_dir is None:
        result_dir = Path.cwd() / "sret_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据预处理
    df_processed = preprocess_sret_data(df, target_blocks, block_col)
    if df_processed is None:
        return {}

    # 2. 基础分析
    basic_results = analyze_basic(df_processed)

    # 3. 词性分析
    if "stim_type" in df_processed.columns:
        valence_results = analyze_valence(df_processed)
    else:
        valence_results = {}
        warnings.warn("未找到stim_type列，跳过词性分析")

    # 4. 计算关键指标
    key_metrics = calculate_key_metrics(df_processed, valence_results)

    # 5. 与图片参考值比较
    comparison_results = compare_with_picture_reference(key_metrics)

    # 6. 生成可视化
    create_visualizations(
        df_processed, basic_results, valence_results, key_metrics, result_dir
    )

    # 7. 保存结果
    results = {
        "basic": basic_results,
        "valence": valence_results,
        "key_metrics": key_metrics,
        "comparison": comparison_results,
    }

    save_results(results, result_dir)

    return results


# ============================================================================
# 运行函数
# ============================================================================


def run_sret_analysis(cfg=None):
    """运行自我参照编码任务分析"""
    print("=" * 60)
    print("自我参照编码任务分析系统")
    print("=" * 60)

    # 获取文件路径
    file_input = input("请输入数据文件路径: ").strip()
    file_path = Path(file_input.strip("'").strip('"')).resolve()

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    # 读取数据
    df = pl.read_csv(file_path)

    # 运行分析

    if cfg is None:
        result_dir = file_path.parent.parent / "results"
    else:
        result_dir = Path(cfg.result_dir)
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
