import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl


@dataclass
class DataUtils:
    date: str = datetime.now().strftime("%Y-%m-%d")
    session_id: int = 0


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
