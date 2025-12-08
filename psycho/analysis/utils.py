from dataclasses import dataclass
from datetime import datetime

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
