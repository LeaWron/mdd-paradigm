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

from psycho.analysis.utils import DataUtils, extract_trials_by_block, find_exp_files

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


def calculate_key_metrics_single(df: pl.DataFrame) -> dict[str, float]:
    """计算单个被试的关键指标"""
    metrics = {}

    total = df.height
    yes_count = df.filter(pl.col("response") == "yes").height

    endorsement_rate = yes_count / total if total > 0 else 0
    metrics["endorsement_rate"] = endorsement_rate

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


def calculate_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    test_type: str = "two_sample",
) -> dict[str, Any]:
    from scipy import stats

    # 计算Z分数
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # 双侧检验
    z_beta = stats.norm.ppf(power)

    # 根据检验类型计算样本量
    if test_type == "one_sample":
        # 单样本t检验
        n = ((z_alpha + z_beta) ** 2) / (effect_size**2)
    elif test_type == "paired":
        # 配对样本t检验
        n = ((z_alpha + z_beta) ** 2) / (effect_size**2)
    elif test_type == "two_sample":
        # 独立样本t检验（每组样本量）
        n_per_group = 2 * ((z_alpha + z_beta) ** 2) / (effect_size**2)
        n_total = 2 * n_per_group
        n = {
            "per_group": n_per_group,
            "total": n_total,
            "per_group_rounded": int(np.ceil(n_per_group)),
            "total_rounded": int(np.ceil(n_total)),
        }
    else:
        raise ValueError(f"不支持的检验类型: {test_type}")

    effect_size_magnitude = ""
    if abs(effect_size) < 0.2:
        effect_size_magnitude = "很小 (very small)"
    elif abs(effect_size) < 0.5:
        effect_size_magnitude = "小 (small)"
    elif abs(effect_size) < 0.8:
        effect_size_magnitude = "中等 (medium)"
    else:
        effect_size_magnitude = "大 (large)"

    return {
        "effect_size": effect_size,
        "effect_size_magnitude": effect_size_magnitude,
        "alpha": alpha,
        "power": power,
        "test_type": test_type,
        "required_n": n if test_type != "two_sample" else n["per_group_rounded"],
        "required_n_total": n["total_rounded"] if test_type == "two_sample" else None,
        "z_alpha": z_alpha,
        "z_beta": z_beta,
        "formula_used": f"n = {2 if test_type == 'two_sample' else 1} * ((Z_α + Z_β)² / d²)",
    }


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


def create_visualizations_single(
    metrics: dict[str, float],
    valence_results: dict[str, Any],
    result_dir: Path,
) -> go.Figure:
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

    return fig


def save_results_single(
    metrics: dict[str, float],
    valence_results: dict[str, Any],
    result_dir: Path,
):
    """保存结果"""

    metrics_df = pl.DataFrame([metrics])
    metrics_df.write_csv(result_dir / "sret_key_metrics.csv")

    if "valence_stats" in valence_results:
        valence_df = pl.DataFrame(valence_results["valence_stats"])
        valence_df.write_csv(result_dir / "sret_valence_stats.csv")

    results = {
        "key_metrics": metrics,
        "valence_analysis": valence_results,
    }

    with open(result_dir / "sret_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {result_dir}")
    print("  - sret_key_metrics.csv (关键指标)")
    print("  - sret_valence_stats.csv (词性统计)")
    print("  - sret_analysis_results.json (完整结果)")
    print("  - sret_analysis_report.html (分析报告)")


def analyze_sret_data_single(
    df: pl.DataFrame,
    target_blocks: list[str] = ["Encoding"],
    result_dir: Path = Path("results"),
) -> dict[str, Any]:
    """分析单个被试的SRET数据"""

    # 1. 提取试次数据
    trials_df = load_and_preprocess_data(df)

    # 2. 计算关键指标
    metrics = calculate_key_metrics_single(trials_df)

    # 3. 词性表现分析
    valence_results = analyze_valence_performance(trials_df)

    # 4. 反应时细分分析
    rt_breakdown = analyze_reaction_time_breakdown(trials_df)

    # 5. 创建可视化
    fig = create_visualizations_single(metrics, valence_results, result_dir)  # noqa: F841

    # 6. 保存结果
    save_results_single(metrics, valence_results, result_dir)

    # 7. 生成报告
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
        "rt_breakdown": rt_breakdown,
    }

    return results


def check_normality_and_homoscedasticity(
    group_metrics: list[dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """检查正态性和方差齐性"""
    results = {}

    # 转换为DataFrame
    df = pd.DataFrame(group_metrics)

    # 需要检验的指标
    key_metrics = [
        "positive_bias",
        "rt_negative_minus_positive",
        "rt_endorsed_minus_not",
        "endorsement_rate",
    ]

    for metric in key_metrics:
        if metric not in df.columns:
            continue

        values = df[metric].dropna().values

        if len(values) < 3:
            results[metric] = {"error": "样本量不足进行正态性检验"}
            continue

        # 正态性检验（Shapiro-Wilk）
        try:
            stat, p_value = stats.shapiro(values)
            is_normal = p_value > 0.05

            # 方差齐性检验
            if len(group_metrics) >= 2:
                levene_stat, levene_p = stats.levene(values, values)
                is_homoscedastic = levene_p > 0.05
            else:
                is_homoscedastic = None

            results[metric] = {
                "shapiro_stat": float(stat),
                "shapiro_p": float(p_value),
                "is_normal": is_normal,
                "levene_stat": float(levene_stat)
                if "levene_stat" in locals()
                else None,
                "levene_p": float(levene_p) if "levene_p" in locals() else None,
                "is_homoscedastic": is_homoscedastic,
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
        except Exception as e:
            results[metric] = {"error": f"检验失败: {str(e)}"}

    return results


def perform_group_comparisons(
    control_metrics: list[dict[str, float]],
    experimental_metrics: list[dict[str, float]],
    anova: bool = True,
) -> dict[str, dict[str, Any]]:
    """对照组和实验组的比较"""
    results = {}

    control_df = pd.DataFrame(control_metrics)
    experimental_df = pd.DataFrame(experimental_metrics)

    key_metrics = [
        "positive_bias",
        "rt_negative_minus_positive",
        "rt_endorsed_minus_not",
        "endorsement_rate",
    ]

    for metric in key_metrics:
        if metric not in control_df.columns or metric not in experimental_df.columns:
            continue

        control_values = control_df[metric].dropna().values
        experimental_values = experimental_df[metric].dropna().values

        if len(control_values) < 2 or len(experimental_values) < 2:
            results[metric] = {"error": "样本量不足进行组间比较"}
            continue

        try:
            # 基础正态性和方差齐性检查
            _, control_p = stats.shapiro(control_values)
            _, experimental_p = stats.shapiro(experimental_values)
            both_normal = control_p > 0.05 and experimental_p > 0.05
            levene_stat, levene_p = stats.levene(control_values, experimental_values)
            equal_var = levene_p > 0.05

            if anova:
                f_stat, p_value = stats.f_oneway(control_values, experimental_values)

                k = 2
                N = len(control_values) + len(experimental_values)
                df_between = k - 1
                df_within = N - k
                degrees_of_freedom = f"{df_between}, {df_within}"

                eta_squared = (f_stat * df_between) / (f_stat * df_between + df_within)

                effect_size = eta_squared
                effect_size_type = "Eta-squared"

                if eta_squared < 0.01:
                    effect_size_desc = "可忽略 (Negligible)"
                elif eta_squared < 0.06:
                    effect_size_desc = "小 (Small)"
                elif eta_squared < 0.14:
                    effect_size_desc = "中等 (Medium)"
                else:
                    effect_size_desc = "大 (Large)"
                effect_size_desc = f"{effect_size_desc} (η²={eta_squared:.3f})"

                test_type = "One-way ANOVA"
                statistic = f_stat
                cohens_d = None

            else:
                degrees_of_freedom = "N/A"
                eta_squared = None

                if both_normal:
                    # 参数检验：独立样本t检验
                    if equal_var:
                        t_stat, p_value = stats.ttest_ind(
                            control_values, experimental_values, equal_var=True
                        )
                        test_type = "Student's t-test (equal variance)"
                        df_t = len(control_values) + len(experimental_values) - 2
                        degrees_of_freedom = str(df_t)
                    else:
                        t_stat, p_value = stats.ttest_ind(
                            control_values, experimental_values, equal_var=False
                        )
                        test_type = "Welch's t-test (unequal variance)"
                        degrees_of_freedom = "Welch approx."

                    statistic = t_stat
                else:
                    # 非参数检验：Mann-Whitney U检验
                    u_stat, p_value = stats.mannwhitneyu(
                        control_values, experimental_values
                    )
                    statistic = u_stat
                    test_type = "Mann-Whitney U test"
                    degrees_of_freedom = "N/A"

                # 计算效应量 (Cohen's d or Cliff's delta concept)
                if both_normal:
                    n1, n2 = len(control_values), len(experimental_values)
                    pooled_std = np.sqrt(
                        (
                            (n1 - 1) * np.var(control_values, ddof=1)
                            + (n2 - 1) * np.var(experimental_values, ddof=1)
                        )
                        / (n1 + n2 - 2)
                    )
                    cohens_d = (
                        np.mean(control_values) - np.mean(experimental_values)
                    ) / pooled_std

                    effect_size = cohens_d
                    effect_size_type = "Cohen's d"

                    abs_d = abs(cohens_d)
                    if abs_d < 0.2:
                        effect_size_desc = "很小"
                    elif abs_d < 0.5:
                        effect_size_desc = "小"
                    elif abs_d < 0.8:
                        effect_size_desc = "中等"
                    else:
                        effect_size_desc = "大"
                    effect_size_desc = f"{effect_size_desc} (d={cohens_d:.2f})"
                else:
                    cohens_d = None
                    effect_size = None
                    effect_size_type = "Non-parametric"
                    effect_size_desc = "非参数效应量"

            sample_size_info = {}
            if cohens_d is not None:
                sample_size_info = calculate_sample_size(
                    effect_size=abs(cohens_d),
                    alpha=0.05,
                    power=0.8,
                    test_type="two_sample",
                )

            results[metric] = {
                "test_type": test_type,
                "statistic": float(statistic) if not np.isnan(statistic) else None,
                "p_value": float(p_value),
                "cohens_d": float(cohens_d) if cohens_d is not None else None,
                "effect_size_desc": effect_size_desc,
                "analysis_type": test_type,
                "degrees_of_freedom": degrees_of_freedom,
                "effect_size": float(effect_size) if effect_size is not None else None,
                "effect_size_type": effect_size_type,
                "effect_size_magnitude": effect_size_desc.split("(")[0].strip()
                if "(" in effect_size_desc
                else effect_size_desc,
                # 样本量信息
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
                # 描述性统计
                "control_mean": float(np.mean(control_values)),
                "control_std": float(np.std(control_values, ddof=1)),
                "control_n": len(control_values),
                "experimental_mean": float(np.mean(experimental_values)),
                "experimental_std": float(np.std(experimental_values, ddof=1)),
                "experimental_n": len(experimental_values),
                "both_normal": both_normal,
                "equal_variance": equal_var if "equal_var" in locals() else None,
            }
        except Exception as e:
            results[metric] = {"error": f"比较分析失败: {str(e)}"}

    return results


def create_group_comparison_visualizations_single_group(
    group_metrics: list[dict[str, float]],
    statistical_results: dict[str, dict[str, Any]],
    result_dir: Path,
):
    """单个组的组分析可视化"""

    all_metrics = group_metrics

    key_metrics_list = [
        "positive_bias",
        "rt_negative_minus_positive",
        "rt_endorsed_minus_not",
        "endorsement_rate",
    ]

    metric_names = ["积极偏向", "消极-积极RT差", "认同-不认同RT差", "总认同率"]

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "1. 各被试积极偏向分布",
            "2. 关键指标相关性",
            "3. 与文献对比",
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

    # 图1: 各被试积极偏向分布
    if "positive_bias" in all_metrics[0]:
        bias_values = [m["positive_bias"] for m in all_metrics]
        subjects = [f"被试{i + 1}" for i in range(len(all_metrics))]

        fig.add_trace(
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

    # 图2: 关键指标相关性热图
    metrics_df = pd.DataFrame(all_metrics)
    available_metrics = [m for m in key_metrics_list if m in metrics_df.columns]
    available_names = [
        metric_names[key_metrics_list.index(m)] for m in available_metrics
    ]

    if available_metrics:
        corr_matrix = metrics_df[available_metrics].corr()

        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=available_names,
                y=available_names,
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
            colorbar=dict(len=0.4, y=0.85),
            selector=dict(type="heatmap"),
        )

    # 图3: 积极偏向对比参考
    if "positive_bias" in all_metrics[0]:
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

        group_mean = np.mean([m["positive_bias"] for m in all_metrics])

        fig.add_trace(
            go.Bar(
                x=["当前组", "对照组", "mdd组"],
                y=[group_mean, hco_bias, mdd_bias],
                name="积极偏向比较",
                marker_color=["blue", "green", "red"],
                text=[f"{group_mean:.3f}", f"{hco_bias:.3f}", f"{mdd_bias:.3f}"],
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
                if "t_statistic" in result:
                    test_data.append(
                        [
                            name,
                            f"{result['t_statistic']:.3f}",
                            f"{result['p_value']:.4f}",
                            f"{result.get('cohens_d', 'N/A')}",
                            f"{result.get('effect_size_desc', 'N/A')}",
                        ]
                    )

        if test_data:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["指标", "t值", "p值", "效应量", "效应大小"],
                        fill_color="lightblue",
                        align="left",
                        font=dict(size=10),
                    ),
                    cells=dict(
                        values=np.array(test_data).T,
                        fill_color="lavender",
                        align="left",
                        font=dict(size=9),
                    ),
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
        if metric in all_metrics[0]:
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
    if "error" not in statistical_results and "positive_bias" in statistical_results:
        # 生成不同效应量下的样本量需求曲线
        effect_sizes = np.linspace(0.1, 1.0, 20)
        sample_sizes = []

        for d in effect_sizes:
            sample_size_info = calculate_sample_size(
                effect_size=d, alpha=0.05, power=0.8, test_type="two_sample"
            )
            sample_sizes.append(sample_size_info["required_n"])

        # 获取当前效应量
        current_d = statistical_results["positive_bias"].get("cohens_d")

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

    # 更新布局
    fig.update_layout(
        title=dict(
            text="SRET组分析报告",
            font=dict(size=22, family="Arial Black"),
            x=0.5,
        ),
        height=1400,
        width=1600,
        showlegend=True,
        template="plotly_white",
    )

    fig.write_html(str(result_dir / "sret_group_analysis_report.html"))

    return fig


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
            "1. 积极偏向分布",
            "2. 消极-积极RT差分布",
            "3. 认同-不认同RT差分布",
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
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
    )

    # 图1-3: 指标分布箱形图
    key_metrics = [
        "positive_bias",
        "rt_negative_minus_positive",
        "rt_endorsed_minus_not",
    ]
    metric_names = ["积极偏向", "消极-积极RT差", "认同-不认同RT差"]

    for i, (metric, name) in enumerate(zip(key_metrics, metric_names)):
        if metric in control_values:
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
    control_means = []
    control_stds = []
    experimental_means = []
    experimental_stds = []
    valid_metrics = []

    for metric, name in zip(key_metrics, metric_names):
        if metric in control_values and metric in experimental_values:
            control_means.append(np.mean(control_values[metric]))
            control_stds.append(np.std(control_values[metric], ddof=1))
            experimental_means.append(np.mean(experimental_values[metric]))
            experimental_stds.append(np.std(experimental_values[metric], ddof=1))
            valid_metrics.append(name)

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
                text=[f"{v:.3f}" for v in control_means],
                textposition="outside",
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
                text=[f"{v:.3f}" for v in experimental_means],
                textposition="outside",
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(ticktext=valid_metrics, tickvals=x_positions, row=2, col=1)

    # 图5: 统计检验结果表格
    if comparison_results:
        table_data = []
        for metric, name in zip(key_metrics, metric_names):
            if metric in comparison_results:
                result = comparison_results[metric]

                # Safe formatting for optional values
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
                        name,
                        str(
                            result.get("analysis_type", result.get("test_type", "N/A"))
                        ),
                        f"{stat_val:.3f}" if stat_val is not None else "N/A",
                        f"{p_val:.4f}" if p_val is not None else "N/A",
                        f"{eff_size:.3f}" if eff_size is not None else "N/A",
                        str(eff_mag),
                    ]
                )

        if table_data:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=[
                            "指标",
                            "检验方法",
                            "统计量",
                            "p值",
                            "效应量",
                            "效应大小",
                        ],
                        fill_color="lightblue",
                        align="left",
                        font=dict(size=10),
                    ),
                    cells=dict(
                        values=np.array(table_data).T,
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

        for metric, name in zip(key_metrics, metric_names):
            if metric in comparison_results:
                # Try new key 'effect_size', fallback to old 'cohens_d'
                val = comparison_results[metric].get("effect_size")
                if val is None:
                    val = comparison_results[metric].get("cohens_d")

                if val is not None:
                    effect_sizes.append(abs(val))
                    metric_labels.append(name)

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
                            f"{result.get('cohens_d', 'N/A'):.3f}",
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

        # 获取当前效应量（使用第一个指标）
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
            text="SRET组间比较分析报告（含样本量计算）",
            font=dict(size=22, family="Arial Black"),
            x=0.5,
        ),
        height=1400,
        width=1600,
        showlegend=True,
        template="plotly_white",
    )

    fig.write_html(str(result_dir / "sret_group_comparison_report.html"))

    return fig


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
    reference_group: Literal["control", "mdd"] = None,
):
    """组SRET分析"""

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

    # 单样本t检验
    statistical_results = {}

    key_metrics = [
        "positive_bias",
        "rt_negative_minus_positive",
        "rt_endorsed_minus_not",
        "endorsement_rate",
    ]

    for metric in key_metrics:
        # 获取当前组的指标值
        group_values = [m[metric] for m in group_metrics if metric in m]

        if len(group_values) < 2:
            statistical_results[metric] = {"error": "样本量不足"}
            continue

        if metric == "positive_bias":
            if reference_group == "control":
                positive_rate = (
                    REFERENCE_VALUES["control"]["endorsement_count"]["positive"] / 40
                )
                negative_rate = (
                    REFERENCE_VALUES["control"]["endorsement_count"]["negative"] / 40
                )
                ref_value = positive_rate - negative_rate
            else:
                positive_rate = (
                    REFERENCE_VALUES["mdd"]["endorsement_count"]["positive"] / 40
                )
                negative_rate = (
                    REFERENCE_VALUES["mdd"]["endorsement_count"]["negative"] / 40
                )
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
            ref_value = 0  # 假设没有差异
        elif metric == "endorsement_rate":
            # 估算总认同率
            if reference_group == "control":
                positive_endorsed = REFERENCE_VALUES["control"]["endorsement_count"][
                    "positive"
                ]
                negative_endorsed = REFERENCE_VALUES["control"]["endorsement_count"][
                    "negative"
                ]
                ref_value = (positive_endorsed + negative_endorsed) / 80
            else:
                positive_endorsed = REFERENCE_VALUES["mdd"]["endorsement_count"][
                    "positive"
                ]
                negative_endorsed = REFERENCE_VALUES["mdd"]["endorsement_count"][
                    "negative"
                ]
                ref_value = (positive_endorsed + negative_endorsed) / 80
        else:
            continue

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
            "reference_mean": float(ref_value),
            "reference_group": reference_group,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "effect_size_desc": effect_size_desc,
            # 样本量信息
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


def run_groups_sret_analysis(
    control_files: list[Path],
    experimental_files: list[Path],
    result_dir: Path = Path("group_comparison_results"),
    groups: list[str] = None,
) -> dict[str, Any]:
    """比较对照组和实验组"""

    control_results = []
    control_metrics = []
    control_name = groups[0] if groups else "control"

    for i, file_path in enumerate(control_files):
        try:
            df = pl.read_csv(file_path)
            subject_id = file_path.stem.split("-")[0]

            subject_result_dir = result_dir / control_name / subject_id
            subject_result_dir.mkdir(parents=True, exist_ok=True)

            result = analyze_sret_data_single(
                df=df,
                target_blocks=["Encoding"],
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
    experimental_name = groups[1] if groups and len(groups) > 1 else "experimental"

    for i, file_path in enumerate(experimental_files):
        try:
            df = pl.read_csv(file_path)
            subject_id = file_path.stem.split("-")[0]

            subject_result_dir = result_dir / experimental_name / subject_id
            subject_result_dir.mkdir(parents=True, exist_ok=True)

            result = analyze_sret_data_single(
                df=df,
                target_blocks=["Encoding"],
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
        control_metrics + experimental_metrics
    )

    print("\n执行组间比较分析...")
    comparison_results = perform_group_comparisons(
        control_metrics, experimental_metrics
    )

    print("\n保存组分析结果...")

    all_control_metrics_df = pd.DataFrame([r["key_metrics"] for r in control_results])
    all_control_metrics_df.insert(0, "group", control_name)
    all_control_metrics_df.insert(
        1, "subject_id", [r["subject_id"] for r in control_results]
    )

    all_experimental_metrics_df = pd.DataFrame(
        [r["key_metrics"] for r in experimental_results]
    )
    all_experimental_metrics_df.insert(0, "group", experimental_name)
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
