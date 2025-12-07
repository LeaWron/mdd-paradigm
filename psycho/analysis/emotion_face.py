import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from psycho.analysis.utils import extract_trials_by_block

warnings.filterwarnings("ignore")

print("=" * 60)
print("面部情绪识别实验数据分析")
print("=" * 60)

# 读取CSV文件
file_path = Path(
    input("请输入面部情绪识别实验数据文件路径:\n").strip("'").strip()
).resolve()
df = pl.read_csv(file_path)

print(f"数据形状: {df.shape}")
print(f"试次总数: {len(df)}")
print(f"数据列: {df.columns}")

target_blocks = [0, 1]  # 定义要分析的区块索引
print(f"目标分析区块: {target_blocks}")

# 调用工具函数，获取清洗和填充后的指定区块数据
df = extract_trials_by_block(
    df=df,
    target_block_indices=target_blocks,
    block_col="block_index",
    trial_col="trial_index",
    fill_na=True,  # 启用空字符串处理
)

print(f"提取后数据形状: {df.shape}")
print(f"新区列 'trial_in_block' 已添加: {'trial_in_block' in df.columns}")

# 检查区块分布
if "block_index" in df.columns:
    block_dist = (
        df.group_by("block_index")
        .agg(pl.count().alias("trial_count"))
        .sort("block_index")
    )
    print("\n提取数据中的区块分布:")
    for row in block_dist.iter_rows():
        print(f"  区块 {row[0]}: {row[1]} 个试次")

# 2. 数据映射与计算列添加 (在已处理数据上进行)
print("\n" + "=" * 60)
print("数据映射与衍生列计算")
print("=" * 60)


# 定义函数映射情绪类型
def map_stim_type(col_value):
    if col_value is None:
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


# 应用映射函数，创建 stim_type 和 choice_type 列
df = df.with_columns(
    [
        pl.col("stim")
        .map_elements(map_stim_type, return_dtype=pl.Utf8)
        .alias("stim_type"),
        pl.col("choice")
        .map_elements(map_stim_type, return_dtype=pl.Utf8)
        .alias("choice_type"),
    ]
)

# 3. 计算关键行为指标 (后续代码与原逻辑一致，但数据源已是处理后的 df)
print("\n" + "=" * 60)
print("关键行为指标计算")
print("=" * 60)

# 计算是否正确
df = df.with_columns((pl.col("stim_type") == pl.col("choice_type")).alias("correct"))

# 分块正确率 (基于已提取的区块)
block_correct = (
    df.group_by("block_index")
    .agg(pl.col("correct").mean().alias("correct_rate"))
    .sort("block_index")
)
print("\n分块正确率:")
for row in block_correct.iter_rows():
    print(f"  区块 {row[0]}: {row[1]:.2%}")

# 按情绪类型统计正确率
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

print("\n按刺激情绪类型的正确率:")
for row in emotion_correct.iter_rows():
    print(f"  {row[0]}: {row[1]:.2%} (N={row[2]})")

# 反应时分析与清理
df = df.with_columns(
    pl.when(pl.col("rt") < 0.1)
    .then(0.1)
    .when(pl.col("rt") > 5.0)
    .then(5.0)
    .otherwise(pl.col("rt"))
    .alias("rt_clean")
)

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

print("\n反应时分析 (秒):")
for row in rt_summary.iter_rows():
    print(f"  {row[0]}: 平均={row[1]:.3f}, 中位数={row[2]:.3f}, SD={row[3]:.3f}")

# 强度一致性分析 (仅对非中性刺激)
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
    print("\n强度评分差异分析:")
    for row in intensity_stats.iter_rows():
        print(f"  {row[0]}: 平均差异={row[1]:.2f}, 绝对差异={row[2]:.2f}")
else:
    print("\n强度评分差异分析: 无有效非中性刺激数据")


# 4. 使用 Plotly 创建交互式图表
print("\n" + "=" * 60)
print("生成交互式可视化图表")
print("=" * 60)

emotion_correct_pd = emotion_correct.to_pandas()
rt_summary_pd = rt_summary.to_pandas()
intensity_stats_pd = intensity_stats.to_pandas()
# 为绘图提取数据样本，也可以选择转换为Pandas
scatter_sample_pd = df.sample(fraction=0.3, seed=42).to_pandas()  # 采样以提升绘图性能
df_intensity_pd = df_intensity.to_pandas()

# 创建包含子图的仪表板
fig = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=(
        "不同情绪类型的识别正确率",
        "不同情绪类型的反应时分布",
        "反应时与正确率的关系",
        "强度评分一致性",
        "强度评分差异分布",
        "分块正确率变化",
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.1,
)

# 4.1 正确率分布 (柱状图)
fig.add_trace(
    go.Bar(
        x=emotion_correct_pd["stim_type"],
        y=emotion_correct_pd["correct_rate"],
        text=[f"{rate:.1%}" for rate in emotion_correct_pd["correct_rate"]],
        textposition="auto",
        marker_color=["#636efa", "#00cc96", "#ef553b"],  # 为不同情绪类型设置颜色
    ),
    row=1,
    col=1,
)
fig.update_yaxes(range=[0, 1.05], title_text="正确率", row=1, col=1)

# 4.2 反应时分布 (箱线图)
# 为每种情绪类型准备数据列表
for stim_type in ["positive", "neutral", "negative"]:
    rt_data = (
        df.filter(pl.col("stim_type") == stim_type)["rt_clean"].drop_nulls().to_list()
    )
    fig.add_trace(
        go.Box(y=rt_data, name=stim_type, marker_color="lightseagreen", boxmean="sd"),
        row=1,
        col=2,
    )
fig.update_yaxes(title_text="反应时 (秒)", row=1, col=2)

# 4.3 正确率与反应时的关系 (散点图)
fig.add_trace(
    go.Scatter(
        x=scatter_sample_pd["rt_clean"],
        y=scatter_sample_pd["correct"].astype(int),
        mode="markers",
        marker=dict(
            size=8,
            color=scatter_sample_pd["stim_type"].map(
                {"positive": 0, "neutral": 1, "negative": 2}
            ),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="情绪类型", tickvals=[0, 1, 2], ticktext=["积极", "中性", "消极"]
            ),
        ),
        text=scatter_sample_pd["stim_type"],
        hovertemplate="<b>反应时</b>: %{x:.3f}s<br><b>是否正确</b>: %{y}<br><b>类型</b>: %{text}<extra></extra>",
    ),
    row=1,
    col=3,
)
fig.update_yaxes(
    tickvals=[0, 1], ticktext=["错误", "正确"], title_text="是否正确", row=1, col=3
)
fig.update_xaxes(title_text="反应时 (秒)", row=1, col=3)

# 4.4 强度评分一致性 (散点图)
fig.add_trace(
    go.Scatter(
        x=df_intensity_pd["label_intensity"],
        y=df_intensity_pd["intensity"],
        mode="markers",
        marker=dict(
            size=10,
            color=df_intensity_pd["stim_type"].map({"positive": 0, "negative": 1}),
            colorscale=["#00cc96", "#ef553b"],
            showscale=True,
            colorbar=dict(title="情绪类型", tickvals=[0, 1], ticktext=["积极", "消极"]),
        ),
        text=df_intensity_pd["stim_type"],
        hovertemplate="<b>标签强度</b>: %{x}<br><b>被试评分</b>: %{y}<br><b>类型</b>: %{text}<extra></extra>",
    ),
    row=2,
    col=1,
)
# 添加对角线 (完美一致性)
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

# 4.5 强度评分差异分布 (箱线图)
for stim_type in ["positive", "negative"]:
    diff_data = (
        df_intensity.filter(pl.col("stim_type") == stim_type)["intensity_diff"]
        .drop_nulls()
        .to_list()
    )
    fig.add_trace(
        go.Box(y=diff_data, name=stim_type, marker_color="orange"), row=2, col=2
    )
# 添加零参考线
fig.add_hline(y=0, line_dash="dot", line_color="red", row=2, col=2)
fig.update_yaxes(title_text="评分差异 (被试评分 - 标签强度)", row=2, col=2)

# 4.6 分块正确率变化 (折线图)
fig.add_trace(
    go.Scatter(
        x=block_correct["block_index"].to_list(),
        y=block_correct["correct_rate"].to_list(),
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
    showlegend=False,
    hovermode="closest",
)

# 保存图表
try:
    fig.write_html("facial_emotion_analysis_interactive.html")
    fig.write_image("facial_emotion_analysis_static.png", scale=2)
    print("交互式图表已保存为 'facial_emotion_analysis_interactive.html'")
    print("静态图片已保存为 'facial_emotion_analysis_static.png'")
except Exception as e:
    print(f"保存图表时出错: {e}")

# 5. 高级统计分析 (需要将数据转换为Pandas以兼容statsmodels)
print("\n" + "=" * 60)
print("高级统计分析")
print("=" * 60)

# 转换为Pandas DataFrame进行统计分析
df_pd = df.to_pandas()

# 5.1 不同情绪类型正确率的卡方检验
print("\n1. 不同情绪类型正确率的差异性检验:")
contingency_table = pd.crosstab(df_pd["stim_type"], df_pd["correct"])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"   卡方检验: χ²({dof}) = {chi2:.3f}, p = {p:.4f}")

# 5.2 反应时的方差分析
print("\n2. 不同情绪类型反应时的ANOVA分析:")
anova_data = df_pd[["stim_type", "rt_clean"]].dropna()
model = ols("rt_clean ~ C(stim_type)", data=anova_data).fit()
anova_table = anova_lm(model)
print(
    f"   ANOVA结果: F({anova_table['df'][0]}, {anova_table['df'][1]}) = {anova_table['F'][0]:.3f}, p = {anova_table['PR(>F)'][0]:.4f}"
)

# 5.3 强度评分相关性分析
print("\n3. 强度评分相关性分析:")
if len(df_intensity_pd) > 10:
    # 选取相关列，并丢弃任何一列中包含NaN的整行
    intensity_corr_data = df_intensity_pd[["label_intensity", "intensity"]].dropna()

    # 再次检查丢弃缺失值后是否还有足够的数据点进行计算
    if len(intensity_corr_data) >= 3:  # Pearson相关至少需要3个点
        x = intensity_corr_data["label_intensity"]
        y = intensity_corr_data["intensity"]
        corr, p_val = stats.pearsonr(x, y)
        print(f"   标签强度与被试评分的相关性: r = {corr:.3f}, p = {p_val:.4f}")
        print(f"   用于计算的有效数据点: {len(intensity_corr_data)} 个")
    else:
        print("   警告: 在同时丢弃缺失值后，有效数据点不足，无法计算相关性。")
        print(f"   剩余数据点: {len(intensity_corr_data)} 个")

# 5.4 速度-准确性权衡分析
print("\n4. 速度-准确性权衡分析:")
df_pd["rt_quartile"] = pd.qcut(
    df_pd["rt_clean"], 4, labels=["最快", "较快", "较慢", "最慢"]
)
speed_accuracy = df_pd.groupby("rt_quartile")["correct"].mean().reset_index()
speed_accuracy.columns = ["反应时分组", "正确率"]

print("   不同反应时分组下的正确率:")
for _, row in speed_accuracy.iterrows():
    print(f"     {row['反应时分组']}: {row['正确率']:.2%}")

# 6. 生成详细报告
print("\n" + "=" * 60)
print("分析报告总结")
print("=" * 60)

overall_accuracy = df["correct"].mean()
median_rt = df["rt_clean"].median()

print(f"总体正确率: {overall_accuracy:.2%}")
print(f"总体中位反应时: {median_rt:.3f} 秒")
print(f"有效试次数: {len(df)}")

# 按情绪类型总结
print("\n分情绪类型总结:")
for stim_type in ["positive", "neutral", "negative"]:
    subset = df.filter(pl.col("stim_type") == stim_type)
    if len(subset) > 0:
        acc = subset["correct"].mean()
        rt_mean = subset["rt_clean"].mean()
        rt_std = subset["rt_clean"].std()
        print(f"  {stim_type}:")
        print(f"    正确率: {acc:.2%}")
        print(f"    平均反应时: {rt_mean:.3f} ± {rt_std:.3f} 秒")
        print(f"    试次数: {len(subset)}")

# 7. 输出到文件
print("\n" + "=" * 60)
print("数据输出")
print("=" * 60)

# 保存处理后的数据 (使用 Polars 写 CSV)
df.write_csv("processed_emotion_data_polars.csv")
print("处理后的数据已保存为 'processed_emotion_data_polars.csv'")

# 保存详细统计结果
summary_stats = pl.DataFrame(
    {
        "指标": ["总体正确率", "总体中位反应时(秒)", "总试次数"],
        "值": [overall_accuracy, median_rt, len(df)],
    }
)
summary_stats.write_csv("experiment_summary_polars.csv")

emotion_stats = emotion_correct.join(
    rt_summary.select(["stim_type", "mean_rt"]), on="stim_type"
)
emotion_stats.write_csv("emotion_type_stats_polars.csv")

print(
    "统计摘要已保存为 'experiment_summary_polars.csv' 和 'emotion_type_stats_polars.csv'"
)

print("\n" + "=" * 60)
print("分析完成!")
print("=" * 60)
