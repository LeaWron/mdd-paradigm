import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.stats.power as smp
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


@dataclass
class DataUtils:
    date: str = datetime.now().strftime("%Y-%m-%d")
    session_id: int = None
    groups: list[str] = None


def extract_trials_by_block(
    df: pl.DataFrame,
    target_block_indices: int | list[int],
    block_col: str = "block_index",
    trial_col: str = "trial_index",
    fill_na: bool = True,
) -> pl.DataFrame:
    # 处理目标区块索引参数
    if isinstance(target_block_indices, int):
        target_indices = [target_block_indices]
    else:
        target_indices = target_block_indices

    # 创建数据副本
    data = df.clone()

    # 处理空字符串
    if fill_na:
        # 对字符串列，将空字符串替换为null
        str_cols = [col for col in data.columns if data[col].dtype == pl.Utf8]
        for str_col in str_cols:
            data = data.with_columns(
                pl.when(pl.col(str_col).str.strip_chars().str.len_chars() == 0)
                .then(None)
                .otherwise(pl.col(str_col))
                .alias(str_col)
            )

    # 向前填充区块索引
    if block_col in data.columns:
        data = data.with_columns(pl.col(block_col).forward_fill())

    # 筛选有试次索引的行
    valid_trials = data.filter(pl.col(trial_col).is_not_null())

    # 筛选指定区块
    if block_col in valid_trials.columns:
        result = valid_trials.filter(pl.col(block_col).is_in(target_indices))
    else:
        print("注意: 数据中未找到区块列，返回所有合法试次")
        result = valid_trials

    # 添加区块内连续试次计数
    if result.height > 0:  # 确保结果不为空
        result = result.with_columns(
            pl.arange(0, result.height).alias("trial_in_block")
        )

    return result


def parse_date_input(date_str: str = None) -> str:
    """
    日期解析函数，支持多种输入格式，统一输出 YYYY-MM-DD 格式

    支持的输入格式:
    - YYYY-MM-DD (标准格式)
    - YYYY/MM/DD
    - DD-MM-YYYY
    - DD/MM/YYYY
    - MMDD (月日，如 1225 表示 12月25日)
    - DD (日，如 25 表示本月25日)
    - YYYY-MM
    - YYYY/MM
    - MM-DD
    - MM/DD
    - 今天/今日/today
    - 昨天/昨日/yesterday
    - 前天/day before yesterday
    - 数字表示的相对日期 (如: 1 表示昨天, 2 表示前天)

    Args:
        date_str: 输入的日期字符串，如果为None则返回今天

    Returns:
        str: YYYY-MM-DD 格式的日期字符串

    Raises:
        ValueError: 当无法解析日期时抛出异常
    """
    # 如果没有输入，默认为今天
    if not date_str or not date_str.strip():
        return datetime.now().strftime("%Y-%m-%d")

    date_str = date_str.strip().lower()

    # 处理特殊关键词
    if date_str in ["今天", "今日", "today"]:
        return datetime.now().strftime("%Y-%m-%d")
    elif date_str in ["昨天", "昨日", "yesterday"]:
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    elif date_str in ["前天", "前日", "day before yesterday"]:
        return (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

    # 处理数字表示的相对日期
    if date_str.isdigit():
        # 如果是1-31之间的数字，认为是本月的某一天
        day_num = int(date_str)
        if 1 <= day_num <= 31:
            now = datetime.now()
            try:
                return datetime(now.year, now.month, day_num).strftime("%Y-%m-%d")
            except ValueError:
                # 如果该月没有这一天（如2月30日），则使用下个月的第一天减去一天
                if now.month == 12:
                    next_month = datetime(now.year + 1, 1, 1)
                else:
                    next_month = datetime(now.year, now.month + 1, 1)
                return (next_month - timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            # 否则认为是相对天数
            return (datetime.now() - timedelta(days=day_num)).strftime("%Y-%m-%d")

    # 处理MMDD格式 (月日，如 1225 表示 12月25日)
    if re.match(r"^\d{3,4}$", date_str) and len(date_str) in [3, 4]:
        # 3位数的情况，如 125 表示 1月25日
        # 4位数的情况，如 1225 表示 12月25日
        if len(date_str) == 3:
            month = int(date_str[0])
            day = int(date_str[1:])
        else:
            month = int(date_str[:2])
            day = int(date_str[2:])

        # 检查月份和日期的有效性
        if 1 <= month <= 12 and 1 <= day <= 31:
            current_year = datetime.now().year
            try:
                return datetime(current_year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                # 日期无效（如2月30日）
                raise ValueError(f"无效的日期: {month}月{day}日")

    # 处理DD格式 (日，如 25 表示本月25日)
    if re.match(r"^\d{1,2}$", date_str) and len(date_str) <= 2:
        day = int(date_str)
        if 1 <= day <= 31:
            now = datetime.now()
            try:
                return datetime(now.year, now.month, day).strftime("%Y-%m-%d")
            except ValueError:
                # 如果该月没有这一天（如2月30日），则使用下个月的第一天减去一天
                if now.month == 12:
                    next_month = datetime(now.year + 1, 1, 1)
                else:
                    next_month = datetime(now.year, now.month + 1, 1)
                return (next_month - timedelta(days=1)).strftime("%Y-%m-%d")

    # 尝试解析各种日期格式
    # 标准格式 YYYY-MM-DD 或 YYYY/MM/DD
    if re.match(r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$", date_str):
        separator = "/" if "/" in date_str else "-"
        parts = date_str.split(separator)
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        return datetime(year, month, day).strftime("%Y-%m-%d")

    # 格式 DD-MM-YYYY 或 DD/MM/YYYY
    if re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{4}$", date_str):
        separator = "/" if "/" in date_str else "-"
        parts = date_str.split(separator)
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        return datetime(year, month, day).strftime("%Y-%m-%d")

    # 格式 YYYY-MM 或 YYYY/MM
    if re.match(r"^\d{4}[/-]\d{1,2}$", date_str):
        separator = "/" if "/" in date_str else "-"
        parts = date_str.split(separator)
        year, month = int(parts[0]), int(parts[1])
        return datetime(year, month, 1).strftime("%Y-%m-%d")

    # 格式 MM-DD 或 MM/DD (默认年份为当前年份)
    if re.match(r"^\d{1,2}[/-]\d{1,2}$", date_str):
        separator = "/" if "/" in date_str else "-"
        parts = date_str.split(separator)
        month, day = int(parts[0]), int(parts[1])
        current_year = datetime.now().year
        return datetime(current_year, month, day).strftime("%Y-%m-%d")

    # 如果以上都无法匹配，尝试使用datetime的解析能力
    try:
        # 尝试几种常见的日期格式
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%m-%d-%Y",
            "%m/%d/%Y",
            "%Y-%m",
            "%Y/%m",
            "%m-%d",
            "%m/%d",
        ]

        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                # 如果只有月份和日期，使用当前年份
                if "%Y" not in fmt:
                    parsed_date = parsed_date.replace(year=datetime.now().year)
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # 最后尝试使用dateutil库（如果安装了的话）
        try:
            from dateutil import parser

            parsed_date = parser.parse(date_str)
            # 如果没有指定年份，使用当前年份
            if parsed_date.year == 1900:  # dateutil默认年份
                parsed_date = parsed_date.replace(year=datetime.now().year)
            return parsed_date.strftime("%Y-%m-%d")
        except ImportError:
            pass

    except Exception:
        pass

    # 如果所有方法都失败，抛出异常
    raise ValueError(f"无法解析日期: {date_str}")


def find_exp_files(data_dir: Path, experiment_type: str):
    """查找指定目录下的指定实验的行为学结果"""
    files = list(data_dir.glob(f"*{experiment_type}.csv"))
    if not files:
        raise FileNotFoundError(
            f"在目录 {data_dir} 下未找到 {experiment_type} 实验的结果文件"
        )
    return files


def check_normality_and_homoscedasticity(
    group_metrics: list[dict[str, float]],
    key_metrics: list[str] = None,
) -> dict[str, dict[str, Any]]:
    """检查正态性和方差齐性"""
    results = {}

    df = pd.DataFrame(group_metrics)

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


def calculate_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    test_type: str = "two_sample",
    effect_size_type: str = "cohens_d",
) -> dict[str, Any]:
    """
    根据效应量计算所需样本量 (使用 statsmodels)
    """
    # 1. 预处理效应量
    # 你的代码里处理 eta_squared 的逻辑必须保留，因为 statsmodels 的 FTestAnovaPower 也是基于 Cohen's f 的
    if effect_size_type == "eta_squared":
        if effect_size >= 1 or effect_size <= 0:
            effect_size_f = 0
        else:
            effect_size_f = np.sqrt(effect_size / (1 - effect_size))
        effect_size_used = effect_size_f
    else:
        effect_size_used = effect_size

    # 处理无效效应量
    if not effect_size_used or effect_size_used <= 0:
        return {
            "required_n": 0,
            "required_n_total": 0,
            "power": power,
            "alpha": alpha,
            "effect_size_magnitude": "Undefined",
        }

    n_res = 0
    n_total = 0

    # 2. 调用 statsmodels 计算
    try:
        if test_type == "two_sample":
            # 独立样本 t 检验
            # ratio=1 表示两组样本量相等
            solver = smp.TTestIndPower()
            n_per_group = solver.solve_power(
                effect_size=effect_size_used,
                nobs1=None,
                alpha=alpha,
                power=power,
                ratio=1.0,
                alternative="two-sided",
            )
            n_res = int(np.ceil(n_per_group))
            n_total = n_res * 2

        elif test_type == "one_sample":
            # 单样本 t 检验
            solver = smp.TTestPower()
            n = solver.solve_power(
                effect_size=effect_size_used,
                nobs=None,
                alpha=alpha,
                power=power,
                alternative="two-sided",
            )
            n_res = int(np.ceil(n))
            n_total = n_res

        elif test_type == "paired":
            # 配对样本 t 检验 (statsmodels 中通常复用 TTestPower，因为差异值的检验等同于单样本)
            solver = smp.TTestPower()
            n = solver.solve_power(
                effect_size=effect_size_used,
                nobs=None,
                alpha=alpha,
                power=power,
                alternative="two-sided",
            )
            n_res = int(np.ceil(n))
            n_total = n_res

        elif test_type == "anova":
            # One-way ANOVA
            # 默认假设 k=2 (两组)，以保持和你之前的逻辑一致。如果是多组，这里应该通过参数传入 k。
            k_groups = 2
            solver = smp.FTestAnovaPower()
            n_per_group = solver.solve_power(
                effect_size=effect_size_used,
                nobs=None,
                alpha=alpha,
                power=power,
                k_groups=k_groups,
            )
            n_res = int(np.ceil(n_per_group))
            n_total = n_res * k_groups

        else:
            print(f"警告: 未知的 test_type '{test_type}'，无法计算样本量")
            n_res = 0
            n_total = 0

    except Exception as e:
        print(f"计算样本量时出错: {e}")
        n_res = 0
        n_total = 0

    # 3. 生成描述文本 (保持你原来的逻辑)
    effect_size_magnitude = ""
    if effect_size_type == "eta_squared":
        if effect_size < 0.01:
            effect_size_magnitude = "可忽略 (Negligible)"
        elif effect_size < 0.06:
            effect_size_magnitude = "小 (Small)"
        elif effect_size < 0.14:
            effect_size_magnitude = "中等 (Medium)"
        else:
            effect_size_magnitude = "大 (Large)"
    else:
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            effect_size_magnitude = "很小 (very small)"
        elif abs_d < 0.5:
            effect_size_magnitude = "小 (small)"
        elif abs_d < 0.8:
            effect_size_magnitude = "中等 (medium)"
        else:
            effect_size_magnitude = "大 (large)"

    # 4. 返回结果
    return {
        "effect_size": effect_size,
        "effect_size_type": effect_size_type,
        "effect_size_magnitude": effect_size_magnitude,
        "alpha": alpha,
        "power": power,
        "test_type": test_type,
        "required_n": n_res,
        "required_n_total": n_total,
        # 移除了 z_alpha 和 z_beta，因为 statsmodels 使用的是更复杂的分布，不再单纯依赖 Z 分数
        "formula_used": "statsmodels (Non-central t/F distribution)",
    }


def perform_group_comparisons(
    control_metrics: list[dict[str, float]],
    experimental_metrics: list[dict[str, float]],
    key_metrics: list[str] = None,
    anova: bool = True,
) -> dict[str, dict[str, Any]]:
    """执行对照组和实验组的比较分析"""
    results = {}

    control_df = pd.DataFrame(control_metrics)
    experimental_df = pd.DataFrame(experimental_metrics)

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
                # One-way ANOVA(k=2)
                f_stat, p_value = stats.f_oneway(control_values, experimental_values)

                k = 2
                N = len(control_values) + len(experimental_values)
                df_between = k - 1
                df_within = N - k
                degrees_of_freedom = f"{df_between}, {df_within}"

                # 效应量计算: Eta-squared (η²)
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

                sample_size_info = calculate_sample_size(
                    effect_size=eta_squared,
                    alpha=0.05,
                    power=0.8,
                    test_type="anova",  # 使用相同的样本量公式
                    effect_size_type="eta_squared",
                )

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

                # 计算效应量 (Cohen's d)
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

                # 计算样本量（仅当有 Cohen's d 时）
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
                "eta_squared": float(eta_squared) if eta_squared is not None else None,
                "effect_size_desc": effect_size_desc,
                "analysis_type": test_type,
                "degrees_of_freedom": degrees_of_freedom,
                "effect_size": float(effect_size) if effect_size is not None else None,
                "effect_size_type": effect_size_type,
                "effect_size_magnitude": effect_size_desc.split("(")[0].strip()
                if "(" in effect_size_desc
                else effect_size_desc,
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


def save_html_report(
    save_dir: Path,
    save_name: str,
    figures: list[go.Figure],
    title: str = "Analysis Report",
    descriptions: list[str] = None,
):
    """
    将多个 Figure 拼接为一个 HTML 文件。
    解决了超宽图表（如 width=2000）导致背景容器宽度不足的问题。
    """
    if descriptions is None:
        descriptions = [""] * len(figures)

    # 确保保存目录存在
    save_dir.mkdir(parents=True, exist_ok=True)

    html_content = [
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <meta charset="utf-8" />
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 40px;
                    background-color: #f8f9fa;
                    color: #333;
                    /* 使用 flex 确保内部容器能够根据内容宽度居中 */
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                .container {{
                    /* 关键修改：使容器宽度根据内部图表自动撑开，而不是被限制在 1600px */
                    display: block;
                    width: fit-content;
                    min-width: 1000px;
                    max-width: 98vw; /* 防止在极窄屏幕上溢出视口 */
                    margin: 0 auto;
                    background: white;
                    padding: 40px;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                    border-radius: 8px;
                }}
                h1 {{
                    text-align: center;
                    color: #2c3e50;
                    margin-bottom: 30px;
                    border-bottom: 2px solid #eaeaea;
                    padding-bottom: 20px;
                    width: 100%;
                }}
                .section {{
                    margin-bottom: 60px;
                    width: 100%;
                }}
                .desc {{
                    font-size: 1.1em;
                    color: #555;
                    margin-bottom: 20px;
                    padding: 12px 20px;
                    border-left: 5px solid #1abc9c;
                    background-color: #f0fdfa;
                    border-radius: 0 4px 4px 0;
                }}
                .plot-wrapper {{
                    display: flex;
                    justify-content: center;
                    width: 100%;
                    overflow-x: auto; /* 若图表超过视口宽度，允许内部滚动 */
                }}
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    color: #aaa;
                    font-size: 0.9em;
                    width: 100%;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
        """
    ]

    for i, fig in enumerate(figures):
        # 仅第一个图包含 plotly.js 核心库以减小文件体积
        include_plotlyjs = "cdn" if i == 0 else False
        plot_html = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)

        desc_html = (
            f'<div class="desc">{descriptions[i]}</div>' if descriptions[i] else ""
        )

        section_html = f"""
        <div class="section">
            {desc_html}
            <div class="plot-wrapper">
                {plot_html}
            </div>
        </div>
        """
        html_content.append(section_html)

    html_content.append(f"""
                <div class="footer">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            </div>
        </body>
        </html>
    """)

    # Write to file
    with open((save_dir / save_name).with_suffix(".html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))

    print(f"报告已生成: {save_dir / save_name}.html")


def create_common_single_group_figures(
    group_metrics: list[dict[str, float]],
    statistical_results: dict[str, dict[str, Any]],
    key_metrics: list[str],
    metric_names: list[str],
) -> list[go.Figure]:
    """
    生成单组分析的通用图表。
    """
    figures = []

    # Figure 1: 统计表格 (统计检验结果 & 样本量计算)
    if statistical_results:
        sorted_data = []
        for metric, name in zip(key_metrics, metric_names):
            if (
                metric in statistical_results
                and "error" not in statistical_results[metric]
            ):
                result = statistical_results[metric]
                effect_size_value = ""
                effect_size_type = ""
                if result.get("cohens_d") is not None:
                    effect_size_value = f"d={result.get('cohens_d'):.3f}"
                    effect_size_type = "Cohen's d"
                elif result.get("eta_squared") is not None:
                    effect_size_value = f"η²={result.get('eta_squared'):.3f}"
                    effect_size_type = "Eta-squared"  # noqa
                else:
                    effect_size_value = "N/A"
                # 统计检验结果

                sorted_data.append(
                    {
                        "name": name,
                        "test_type": f"{result.get('test_type', 'N/A')}",
                        "statistic": f"{result.get('statistic', 'N/A'):.3f}",
                        "p_value": f"{result.get('p_value', 'N/A'):.4f}",
                        "effect_size_type": effect_size_type,
                        "effect_size_value": effect_size_value,
                        "effect_size_desc": f"{result.get('effect_size_desc', 'N/A')}",
                    }
                )

                effect_size_value = 0
                # 样本量计算结果
                if (
                    "required_sample_size_per_group" in result
                    and result["required_sample_size_per_group"]
                ):
                    if result.get("cohens_d") is not None:
                        effect_size_value = result.get("cohens_d")
                    elif result.get("eta_squared") is not None:
                        effect_size_value = result.get("eta_squared")

                sorted_data[-1].update(
                    {
                        "sort_effect_size_value": effect_size_value,
                        "required_sample_size_per_group": f"{result['required_sample_size_per_group']}",
                        "required_total_sample_size": f"{result.get('required_total_sample_size', 'N/A')}",
                    }
                )

    sorted_data.sort(key=lambda x: (x["p_value"], abs(x["sort_effect_size_value"])))
    test_data = [
        [
            item["name"],
            item["test_type"],
            item["statistic"],
            item["p_value"],
            item["effect_size_type"],
            item["effect_size_value"],
            item["effect_size_desc"],
        ]
        for item in sorted_data
    ]

    sample_size_data = [
        [
            item["name"],
            item["effect_size_value"],
            item["required_sample_size_per_group"],
            item["required_total_sample_size"],
        ]
        for item in sorted_data
        if item.get("required_sample_size_per_group")
    ]

    fig_tables = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("统计检验结果", "建议样本量计算"),
        specs=[[{"type": "table"}], [{"type": "table"}]],
        vertical_spacing=0.1,
    )
    if test_data:
        fig_tables.add_trace(
            go.Table(
                header=dict(
                    values=[
                        "指标",
                        "检验方法",
                        "统计量",
                        "p值",
                        "效应量类型",
                        "效应量",
                        "效应大小",
                    ],
                    fill_color="lightblue",
                    align="left",
                ),
                cells=dict(
                    values=np.array(test_data).T,
                    fill_color="lavender",
                ),
                # columnwidth=[0.18, 0.18, 0.15, 0.12, 0.15, 0.15, 0.22],
            ),
            row=1,
            col=1,
        )

    if sample_size_data:
        fig_tables.add_trace(
            go.Table(
                header=dict(
                    values=["指标", "效应量", "每组需样本量", "总需样本量"],
                    fill_color="lightcoral",
                ),
                cells=dict(
                    values=np.array(sample_size_data).T,
                    fill_color="mistyrose",
                ),
            ),
            row=2,
            col=1,
        )

    fig_tables.update_layout(height=600, width=1600, title_text="单组效应量-样本量表格")
    figures.append(fig_tables)

    # Figure 2: 图表分析 (效应量分析 & 样本量需求曲线)
    fig_charts = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("关键指标相关性", "样本量需求曲线", "效应量分析"),
        specs=[[{"type": "heatmap"}, {"type": "scatter"}, {"type": "bar"}]],
    )

    # 关键指标相关性
    metrics_df = pd.DataFrame(group_metrics)
    available_metrics = [m for m in key_metrics if m in metrics_df.columns]

    if len(available_metrics) > 1:
        corr_matrix = metrics_df[available_metrics].corr()
        available_names = [
            metric_names[key_metrics.index(m)] for m in available_metrics
        ]

        fig_charts.add_trace(
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
            col=1,
        )
        fig_charts.update_traces(
            colorbar=dict(len=0.6, y=0.25),
            selector=dict(type="heatmap"),
        )

    # 效应量 Bar
    if statistical_results:
        upper_bonud = 1.0
        lower_bound = 0.1

        sorted_data = []

        for metric, name in zip(key_metrics, metric_names):
            if metric in statistical_results:
                result = statistical_results[metric]
                # 效应量分析
                if result.get("eta_squared") is not None:
                    sorted_data.append(
                        {
                            "name": name.split("(")[0].strip(),
                            "effect_size_type": "η²",
                            "effect_size_value": result["eta_squared"],
                            "hover_text": f"{name}<br>η² = {result['eta_squared']:.3f}",
                        }
                    )
                elif result.get("cohens_d") is not None:
                    sorted_data.append(
                        {
                            "name": name.split("(")[0].strip(),
                            "effect_size_type": "d",
                            "effect_size_value": abs(result["cohens_d"]),
                            "hover_text": f"{name}<br>Cohen's d = {abs(result['cohens_d']):.3f}",
                        }
                    )
                    upper_bonud = max(upper_bonud, abs(result["cohens_d"]))
                lower_bound = min(
                    lower_bound, abs(sorted_data[-1]["effect_size_value"])
                )

                # 计算样本量
                effect_size = sorted_data[-1]["effect_size_value"]
                effect_type = sorted_data[-1]["effect_size_type"]

                if effect_type == "η²":
                    sample_size = calculate_sample_size(
                        effect_size,
                        test_type="one_sample",
                        effect_size_type="eta_squared",
                    )["required_n"]
                elif effect_type == "d":
                    sample_size = calculate_sample_size(
                        effect_size, test_type="one_sample"
                    )["required_n"]
                sorted_data[-1]["hover_text"] += f"<br>样本量 = {sample_size:.0f}"
                sorted_data[-1].update({"sample_size": sample_size})

        # 样本量需求曲线
        if len(sorted_data):
            # Cohen's d && η²
            d_effect_sizes = np.linspace(lower_bound, upper_bonud, 100)

            d_sample_sizes = []
            # eta_sample_sizes = []
            for d_effect_size in d_effect_sizes:
                sample_size = calculate_sample_size(
                    d_effect_size, test_type="one_sample"
                )["required_n"]
                d_sample_sizes.append(sample_size)

                # 单组不太会有eta这种情况

            fig_charts.add_trace(
                go.Scatter(
                    x=d_effect_sizes,
                    y=d_sample_sizes,
                    mode="lines",
                    name="样本量需求曲线(Cohen's d)",
                    line=dict(width=3, color="blue"),
                    fill="tozeroy",
                    fillcolor="rgba(255, 0, 0, 0.2)",
                ),
                row=1,
                col=2,
            )

    sorted_data.sort(key=lambda x: x["effect_size_value"])
    effect_sizes = [d["effect_size_value"] for d in sorted_data]
    sample_sizes = [d["sample_size"] for d in sorted_data]
    metrics_names = [d["name"] for d in sorted_data]
    # effect_size_types = [d["effect_size_type"] for d in sorted_data]
    hover_texts = [d["hover_text"] for d in sorted_data]
    if effect_sizes:
        fig_charts.add_trace(
            go.Scatter(
                x=effect_sizes,
                y=sample_sizes,
                mode="markers",
                name="效应量",
                hovertext=hover_texts,
                hoverinfo="text",
                marker=dict(size=15, color="red", symbol="diamond"),
            ),
            row=1,
            col=2,
        )
        fig_charts.add_trace(
            go.Bar(
                x=metrics_names,
                y=effect_sizes,
                name="效应量",
                marker_color="lightgreen",
                text=[f"{v:.3f}" for v in effect_sizes],
                textposition="auto",
                hovertext=hover_texts,
                hoverinfo="text",
            ),
            row=1,
            col=3,
        )

    fig_charts.update_xaxes(title_text="效应量", row=1, col=2)
    fig_charts.update_yaxes(title_text="所需样本量", row=1, col=2)

    fig_charts.update_layout(height=500, width=1600, title_text="单组效应量-样本量图表")
    figures.append(fig_charts)

    return figures


def create_common_comparison_figures(
    comparison_results: dict[str, dict[str, Any]],
    key_metrics: list[str],
    metric_names: list[str],
) -> list[go.Figure]:
    """
    生成组间比较分析的通用图表。
    """
    figures = []

    # Figure 1: 统计表格 (统计检验结果 & 样本量计算)
    if comparison_results:
        sorted_data = []
        for metric, name in zip(key_metrics, metric_names):
            if metric in comparison_results:
                result = comparison_results[metric]

                # 安全格式化可选值
                stat_val = result.get("statistic", "N/A")
                p_val = result.get("p_value", "N/A")
                effect_size = ""
                effect_type = ""
                if result.get("cohens_d") is not None:
                    effect_size = f"d={result.get('cohens_d'):.3f}"
                    effect_type = "Cohen's d"
                elif result.get("eta_squared") is not None:
                    effect_size = f"η²={result.get('eta_squared'):.3f}"
                    effect_type = "η²"
                else:
                    effect_size = "N/A"
                    effect_type = "N/A"

                eff_mag = result.get("effect_size_desc", "N/A")

                sorted_data.append(
                    {
                        "name": name,
                        "test_type": f"{result.get('test_type', 'N/A')}",
                        "statistic": f"{stat_val:.3f}",
                        "p_value": f"{p_val:.4f}",
                        "effect_type": effect_type,
                        "effect_size": effect_size,
                        "effect_size_desc": f"{eff_mag}",
                    }
                )

                if (
                    "required_sample_size_per_group" in result
                    and result["required_sample_size_per_group"]
                ):
                    effect_size = ""
                    effect_type = ""
                    if result.get("cohens_d") is not None:
                        effect_size = result.get("cohens_d")
                        effect_type = "Cohen's d"
                    elif result.get("eta_squared") is not None:
                        effect_size = result.get("eta_squared")
                        effect_type = "η²"

                    sorted_data[-1].update(
                        {
                            "sort_effect_size": effect_size,
                            "required_sample_size_per_group": f"{result['required_sample_size_per_group']}",
                            "required_total_sample_size": f"{result.get('required_total_sample_size', 'N/A')}",
                            "sample_size_power": f"{result.get('sample_size_power', 0.8):.2f}",
                            "sample_size_alpha": f"{result.get('sample_size_alpha', 0.05):.3f}",
                        }
                    )

    sorted_data.sort(key=lambda x: (x["p_value"], abs(x["sort_effect_size"])))
    table_data = [
        [
            item["name"],
            item["test_type"],
            item["statistic"],
            item["p_value"],
            item["effect_type"],
            item["effect_size"],
            item["effect_size_desc"],
        ]
        for item in sorted_data
    ]

    sample_size_data = [
        [
            item["name"],
            item["effect_type"],
            item["effect_size"],
            item["required_sample_size_per_group"],
            item["required_total_sample_size"],
            item["sample_size_power"],
            item["sample_size_alpha"],
        ]
        for item in sorted_data
    ]
    fig_tables = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("组间比较 - 统计检验结果", "组间比较 - 建议样本量"),
        specs=[[{"type": "table"}], [{"type": "table"}]],
        vertical_spacing=0.1,
    )

    if table_data:
        fig_tables.add_trace(
            go.Table(
                header=dict(
                    values=[
                        "指标",
                        "检验方法",
                        "统计量",
                        "P值",
                        "效应量类型",
                        "效应量",
                        "效应量描述",
                    ],
                    fill_color="lightblue",
                ),
                cells=dict(values=np.array(table_data).T, fill_color="lavender"),
            ),
            row=1,
            col=1,
        )

    if sample_size_data:
        fig_tables.add_trace(
            go.Table(
                header=dict(
                    values=[
                        "指标",
                        "效应量类型",
                        "效应量",
                        "单组所需样本量",
                        "总样本量",
                        "统计功效",
                        "显著性水平",
                    ],
                    fill_color="lightcoral",
                ),
                cells=dict(values=np.array(sample_size_data).T, fill_color="mistyrose"),
            ),
            row=2,
            col=1,
        )

    fig_tables.update_layout(
        height=600, width=1600, title_text="组间统计数据-样本量表格"
    )
    figures.append(fig_tables)

    # Figure 2: 图表分析 (样本量需求曲线 & 效应量vs样本量)
    fig_charts = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("样本量需求曲线", "效应量"),
        specs=[[{"type": "scatter"}, {"type": "bar"}]],
    )

    if comparison_results:
        upper_bonud = 1.0
        lower_bound = 0.1

        sorted_data = []

        for metric, name in zip(key_metrics, metric_names):
            if metric in comparison_results:
                result = comparison_results[metric]
                if result.get("eta_squared") is not None:
                    # effect_sizes.append(abs(result.get("eta_squared")))
                    # metrics_names.append(name)
                    # effect_size_types.append("η²")
                    # hover_texts.append(f"{name}<br>η²={result.get('eta_squared'):.3f}")
                    sorted_data.append(
                        {
                            "name": name,
                            "effect_size_value": abs(result.get("eta_squared")),
                            "effect_size_type": "η²",
                            "hover_text": f"{name}<br>η²={result.get('eta_squared'):.3f}",
                        }
                    )
                elif result.get("cohens_d") is not None:
                    # effect_sizes.append(abs(result.get("cohens_d")))
                    # metrics_names.append(name)
                    # effect_size_types.append("Cohen's d")
                    # hover_texts.append(
                    #     f"{name}<br>Cohen's d={result.get('cohens_d'):.3f}"
                    # )
                    sorted_data.append(
                        {
                            "name": name,
                            "effect_size_value": abs(result.get("cohens_d")),
                            "effect_size_type": "Cohen's d",
                            "hover_text": f"{name}<br>Cohen's d={result.get('cohens_d'):.3f}",
                        }
                    )
                    upper_bonud = max(upper_bonud, abs(result.get("cohens_d")))
                lower_bound = min(
                    lower_bound, abs(sorted_data[-1]["effect_size_value"])
                )

                # 计算样本量
                effect_size = sorted_data[-1]["effect_size_value"]
                effect_type = sorted_data[-1]["effect_size_type"]

                if effect_type == "η²":
                    sample_size = calculate_sample_size(
                        effect_size,
                        test_type="anova",
                        effect_size_type="eta_squared",
                    )["required_n"]
                elif effect_type == "d":
                    sample_size = calculate_sample_size(
                        effect_size, test_type="two_sample"
                    )["required_n"]
                sorted_data[-1]["hover_text"] += f"<br>样本量={sample_size:.0f}"
                sorted_data[-1].update({"sample_size": sample_size})

        # 样本量需求曲线
        if len(sorted_data):
            # Cohen's d && η²
            d_effect_sizes = np.linspace(lower_bound, upper_bonud, 100)

            # d_sample_sizes = []
            eta_sample_sizes = []
            for d_effect_size in d_effect_sizes:
                # 对比一般都用 η²

                if d_effect_size <= 1.0:
                    sample_size = calculate_sample_size(
                        d_effect_size,
                        test_type="anova",
                        effect_size_type="eta_squared",
                    )["required_n"]
                    eta_sample_sizes.append(sample_size)

            fig_charts.add_trace(
                go.Scatter(
                    x=d_effect_sizes,
                    y=eta_sample_sizes,
                    mode="lines",
                    name="样本量需求曲线(η²)",
                    line=dict(width=3, color="blue"),
                    fill="tozeroy",
                    fillcolor="rgba(0, 0, 255, 0.2)",
                ),
                row=1,
                col=1,
            )

    sorted_data.sort(key=lambda x: abs(x["effect_size_value"]))

    effect_sizes = [d["effect_size_value"] for d in sorted_data]
    sample_sizes = [d["sample_size"] for d in sorted_data]
    metrics_names = [d["name"] for d in sorted_data]
    hover_texts = [d["hover_text"] for d in sorted_data]

    if effect_sizes:
        fig_charts.add_trace(
            go.Bar(
                x=metrics_names,
                y=effect_sizes,
                name="效应量",
                marker_color="lightgreen",
                text=[f"{v:.3f}" for v in effect_sizes],
                textposition="auto",
                hovertext=hover_texts,
                hoverinfo="text",
            ),
            row=1,
            col=2,
        )

        fig_charts.add_trace(
            go.Scatter(
                x=effect_sizes,
                y=sample_sizes,
                mode="markers",
                name="效应量",
                marker=dict(size=15, color="red", symbol="diamond"),
                hovertext=hover_texts,
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    fig_charts.update_xaxes(title_text="效应量", row=1, col=1)
    fig_charts.update_yaxes(title_text="所需样本量", row=1, col=1)

    fig_charts.update_layout(height=500, width=1600, title_text="组间效应量-样本量图表")
    figures.append(fig_charts)

    return figures


def draw_ci_scatter(
    x: list[float],
    y: list[float],
    y_lower: list[float],
    y_upper: list[float],
    name: str,
    width: int = 3,
    opacity: float = 0.2,
    color: str = "blue",
) -> list[go.Scatter]:
    """
    绘制置信区间散点图

    参数:
        x (list[float]): x轴数据
        y (list[float]): y轴数据
        y_lower (list[float]): y轴下置信区间
        y_upper (list[float]): y轴上置信区间
        name (str): 图例名称
        width (int, optional): 折线宽度. 默认 3.
        opacity (float, optional): 填充透明度. 默认 0.2.
        color (str, optional): 颜色. 默认 "blue".

    返回:
        list[go.Scatter]: 包含下界线、上界填充线和主折线的 Plotly 图表对象列表
    """

    # 常用颜色名到 RGB 的映射 (CSS 标准颜色)
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "magenta": (255, 0, 255),
        "cyan": (0, 255, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203),
        "teal": (0, 128, 128),
        "gold": (255, 215, 0),
    }

    # 解析颜色以生成 rgba 字符串
    if color.startswith("#"):
        # 处理 Hex
        hex_c = color.lstrip("#")
        rgb = tuple(int(hex_c[i : i + 2], 16) for i in (0, 2, 4))
    else:
        # 处理字符串名称 (如果不在字典里，默认返回一个灰色)
        rgb = color_map.get(color.lower(), (128, 128, 128))

    fill_color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"

    # 1. 下界线
    lower_trace = go.Scatter(
        x=x,
        y=y_lower,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )

    # 2. 上界填充线
    upper_trace = go.Scatter(
        x=x,
        y=y_upper,
        mode="lines",
        name=f"{name} ci",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=fill_color,
        showlegend=True,
        hoverinfo="skip",
    )

    # 3. 主折线
    main_trace = go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=width),
        marker=dict(size=8, color=color),
    )

    return [lower_trace, upper_trace, main_trace]
