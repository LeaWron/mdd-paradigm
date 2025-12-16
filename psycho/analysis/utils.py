import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
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
    根据效应量计算所需样本量
    """
    # 计算Z分数
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # 双侧检验
    z_beta = stats.norm.ppf(power)

    # 对于eta_squared，需要转换为Cohen's f
    if effect_size_type == "eta_squared":
        # η² 转换为 Cohen's f: f = sqrt(η²/(1-η²))
        if effect_size >= 1 or effect_size <= 0:
            effect_size_f = 0
        else:
            effect_size_f = np.sqrt(effect_size / (1 - effect_size))
        effect_size_used = effect_size_f
    else:
        effect_size_used = effect_size

    # 根据检验类型计算样本量
    if test_type == "one_sample":
        # 单样本t检验
        n = ((z_alpha + z_beta) ** 2) / (effect_size_used**2)
    elif test_type == "paired":
        # 配对样本t检验
        n = ((z_alpha + z_beta) ** 2) / (effect_size_used**2)
    elif test_type == "two_sample":
        # 独立样本t检验（每组样本量）
        n_per_group = 2 * ((z_alpha + z_beta) ** 2) / (effect_size_used**2)
        n_total = 2 * n_per_group
        n = {
            "per_group": n_per_group,
            "total": n_total,
            "per_group_rounded": int(np.ceil(n_per_group)),
            "total_rounded": int(np.ceil(n_total)),
        }
    elif test_type == "anova":
        # 对于ANOVA，使用Cohen's f计算样本量
        # 公式：n = (φ²/k) + 1，其中 φ = f * sqrt(n)
        # 简化公式：n = ((z_alpha + z_beta)² / f²) * (k/(k-1))
        # 这里我们假设两组比较
        k = 2  # 组数
        n_per_group = ((z_alpha + z_beta) ** 2) / (effect_size_used**2)
        n_total = n_per_group * k
        n = {
            "per_group": n_per_group,
            "total": n_total,
            "per_group_rounded": int(np.ceil(n_per_group)),
            "total_rounded": int(np.ceil(n_total)),
        }
    else:
        raise ValueError(f"不支持的检验类型: {test_type}")

    effect_size_magnitude = ""
    if effect_size_type == "eta_squared":
        # η² 的效应量标准
        if effect_size < 0.01:
            effect_size_magnitude = "可忽略 (Negligible)"
        elif effect_size < 0.06:
            effect_size_magnitude = "小 (Small)"
        elif effect_size < 0.14:
            effect_size_magnitude = "中等 (Medium)"
        else:
            effect_size_magnitude = "大 (Large)"
    else:
        # Cohen's d 的效应量标准
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
        "effect_size_type": effect_size_type,
        "effect_size_magnitude": effect_size_magnitude,
        "alpha": alpha,
        "power": power,
        "test_type": test_type,
        "required_n": n
        if test_type != "two_sample" and test_type != "anova"
        else n["per_group_rounded"],
        "required_n_total": n["total_rounded"]
        if test_type in ["two_sample", "anova"]
        else None,
        "z_alpha": z_alpha,
        "z_beta": z_beta,
        "formula_used": f"n = {2 if test_type in ['two_sample', 'anova'] else 1} * ((Z_α + Z_β)² / d²)",
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

                # 计算基于 η² 的样本量
                # 对于 ANOVA，需要将 η² 转换为 Cohen's f
                if eta_squared >= 1 or eta_squared <= 0:
                    effect_size_for_sample = 0
                else:
                    effect_size_for_sample = np.sqrt(eta_squared / (1 - eta_squared))

                sample_size_info = calculate_sample_size(
                    effect_size=effect_size_for_sample,
                    alpha=0.05,
                    power=0.8,
                    test_type="two_sample",  # 使用相同的样本量公式
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
