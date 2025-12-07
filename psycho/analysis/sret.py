# 认同率: 认同数量 / 总数量
# 偏向: 积极认同率 - 消极认同率
# 消极RT - 积极RT
# 认同RT - 不认同RT
# 消极intensity - 积极intensity
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import polars as pl
from scipy import stats

from psycho.analysis.utils import extract_trials_by_block


# 主分析函数
def analyze_self_reference_data(df, target_blocks=None, block_col="block_index"):
    print("=" * 60)

    # 1. 如果指定了目标区块，提取该区块的数据
    if target_blocks is not None:
        print(f"提取区块 {target_blocks} 的数据...")
        df_analysis = extract_trials_by_block(
            df,
            target_block_indices=target_blocks,
            block_col=block_col,
            fill_na=True,
        )
        print(f"提取后数据形状: {df_analysis.shape}")
    else:
        df_analysis = df.clone()
        print(f"使用所有数据，形状: {df_analysis.shape}")

    # 2. 基础信息
    print("数据基本信息:")
    print(f"总行数: {df_analysis.height}")
    print(f"列数: {df_analysis.width}")
    print(f"列名: {', '.join(df_analysis.columns)}")

    # 检查必要的列是否存在
    required_columns = ["stim_word", "response", "rt", "intensity"]
    missing_columns = [
        col for col in required_columns if col not in df_analysis.columns
    ]
    if missing_columns:
        print(f"️  警告: 缺少必要列 {missing_columns}")
        return None

    # 3. 基础统计计算
    print("\n 基础统计:")

    # 总试次和Yes比例
    total_trials = df_analysis.height
    yes_count = df_analysis.filter(pl.col("response") == "yes").height
    yes_proportion = yes_count / total_trials if total_trials > 0 else 0

    print(f"总试次数: {total_trials}")
    print(f"Yes反应数: {yes_count}")
    print(f"Yes比例: {yes_proportion:.2%}")

    # 反应时统计 (转换为毫秒)
    df_analysis = df_analysis.with_columns((pl.col("rt") * 1000).alias("rt_ms"))

    rt_stats = df_analysis.select(
        [
            pl.col("rt_ms").mean().alias("mean_rt"),
            pl.col("rt_ms").std().alias("std_rt"),
            pl.col("rt_ms").median().alias("median_rt"),
            pl.col("rt_ms").min().alias("min_rt"),
            pl.col("rt_ms").max().alias("max_rt"),
        ]
    )

    print("\n️ 反应时统计 (毫秒):")
    print(f"均值: {rt_stats['mean_rt'][0]:.1f} ms")
    print(f"标准差: {rt_stats['std_rt'][0]:.1f} ms")
    print(f"中位数: {rt_stats['median_rt'][0]:.1f} ms")
    print(f"范围: {rt_stats['min_rt'][0]:.0f} - {rt_stats['max_rt'][0]:.0f} ms")

    # 4. 按反应类型分组分析
    response_stats = (
        df_analysis.group_by("response")
        .agg(
            [
                pl.col("rt_ms").mean().alias("mean_rt"),
                pl.col("rt_ms").std().alias("std_rt"),
                pl.col("intensity").mean().alias("mean_intensity"),
                pl.col("intensity").std().alias("std_intensity"),
                pl.count().alias("count"),
            ]
        )
        .sort("response")
    )

    print("\n 按反应类型分组统计:")
    print(response_stats)

    # 5. 强度评分分析
    print("\n 强度评分分析:")
    intensity_stats = df_analysis.select(
        [
            pl.col("intensity").mean().alias("mean_intensity"),
            pl.col("intensity").std().alias("std_intensity"),
            pl.col("intensity").min().alias("min_intensity"),
            pl.col("intensity").max().alias("max_intensity"),
            pl.col("intensity").median().alias("median_intensity"),
        ]
    )

    print(
        f"整体强度评分: {intensity_stats['mean_intensity'][0]:.2f} ± {intensity_stats['std_intensity'][0]:.2f}"
    )

    # 6. 与Rogers文献数据比较
    rogers_ref = {
        "mean_rt": 2941,  # 毫秒
        "yes_rt": 3194,
        "no_rt": 2689,
        "yes_proportion": 0.613,  # 61.3%
        "recall_rate": 0.32,  # 校正后的回忆率
    }

    # 提取当前数据
    your_data = {
        "mean_rt": rt_stats["mean_rt"][0],
        "yes_rt": response_stats.filter(pl.col("response") == "yes")["mean_rt"][0],
        "no_rt": response_stats.filter(pl.col("response") == "no")["mean_rt"][0],
        "yes_proportion": yes_proportion,
    }

    print("\n📚 与 Rogers et al. (1977) 比较:")
    print(f"平均RT: {your_data['mean_rt']:.0f} ms | 文献: {rogers_ref['mean_rt']} ms")
    print(
        f"差异: {(your_data['mean_rt'] - rogers_ref['mean_rt']):+.0f} ms ({((your_data['mean_rt'] / rogers_ref['mean_rt']) - 1):+.1%})"
    )
    print(
        f"\nYes反应RT: {your_data['yes_rt']:.0f} ms | 文献: {rogers_ref['yes_rt']} ms"
    )
    print(f"No反应RT: {your_data['no_rt']:.0f} ms | 文献: {rogers_ref['no_rt']} ms")
    print(
        f"\nYes比例: {your_data['yes_proportion']:.1%} | 文献: {rogers_ref['yes_proportion']:.1%}"
    )

    # 7. 统计检验
    print("\n📊 统计检验:")

    # Yes vs No反应时差异
    yes_rt = df_analysis.filter(pl.col("response") == "yes")["rt_ms"].to_numpy()
    no_rt = df_analysis.filter(pl.col("response") == "no")["rt_ms"].to_numpy()

    if len(yes_rt) > 1 and len(no_rt) > 1:
        t_stat_rt, p_value_rt = stats.ttest_ind(yes_rt, no_rt, equal_var=False)
        print(f"反应时差异 (Yes vs No): t = {t_stat_rt:.2f}, p = {p_value_rt:.4f}")

        # 效应量 (Cohen's d)
        pooled_std = np.sqrt((np.var(yes_rt, ddof=1) + np.var(no_rt, ddof=1)) / 2)
        cohens_d = (np.mean(yes_rt) - np.mean(no_rt)) / pooled_std
        print(f"效应量 (Cohen's d): {cohens_d:.2f}")
    else:
        print("样本量不足进行反应时t检验")

    # Yes vs No强度差异
    yes_intensity = df_analysis.filter(pl.col("response") == "yes")[
        "intensity"
    ].to_numpy()
    no_intensity = df_analysis.filter(pl.col("response") == "no")[
        "intensity"
    ].to_numpy()

    if len(yes_intensity) > 1 and len(no_intensity) > 1:
        t_stat_int, p_value_int = stats.ttest_ind(
            yes_intensity, no_intensity, equal_var=False
        )
        print(f"强度差异 (Yes vs No): t = {t_stat_int:.2f}, p = {p_value_int:.4f}")
    else:
        print("样本量不足进行强度t检验")

    # 8. 相关性分析
    print("\n 相关性分析:")

    # 整体RT与强度相关
    overall_corr = df_analysis.select(
        [pl.corr("rt_ms", "intensity").alias("correlation")]
    )
    print(f"整体RT-强度相关: r = {overall_corr['correlation'][0]:.3f}")

    # 分反应类型的相关
    for resp in ["yes", "no"]:
        subset = df_analysis.filter(pl.col("response") == resp)
        if subset.height > 2:
            corr = subset.select([pl.corr("rt_ms", "intensity").alias("correlation")])
            print(
                f"{resp.capitalize()}反应RT-强度相关: r = {corr['correlation'][0]:.3f}"
            )

    # 9. 词性分析（如果存在stim_type列）
    if "stim_type" in df_analysis.columns:
        print("\n 词性分析:")
        stim_type_stats = (
            df_analysis.group_by("stim_type")
            .agg(
                [
                    pl.col("response")
                    .filter(pl.col("response") == "yes")
                    .count()
                    .alias("yes_count"),
                    pl.col("response").count().alias("total_count"),
                    pl.col("rt_ms").mean().alias("mean_rt"),
                    pl.col("intensity").mean().alias("mean_intensity"),
                ]
            )
            .with_columns(
                [(pl.col("yes_count") / pl.col("total_count")).alias("yes_proportion")]
            )
        )

        print(stim_type_stats)

    # 10. 实验有效性评估
    print("\n" + "=" * 60)
    print(" 实验有效性评估")
    print("=" * 60)

    # 评估标准
    criteria = {
        "rt_within_range": rogers_ref["mean_rt"] * 0.8
        < your_data["mean_rt"]
        < rogers_ref["mean_rt"] * 1.2,
        "yes_rt_greater": your_data["yes_rt"] > your_data["no_rt"],
        "yes_proportion_reasonable": 0.3 < your_data["yes_proportion"] < 0.8,
        "yes_intensity_greater": None,
    }

    # 检查强度差异
    if len(yes_intensity) > 0 and len(no_intensity) > 0:
        criteria["yes_intensity_greater"] = np.mean(yes_intensity) > np.mean(
            no_intensity
        )

    print("\n评估标准:")
    print(
        "1. 反应时在合理范围内 (2941±20%): ",
        f"{'✓' if criteria['rt_within_range'] else '✗'} ({your_data['mean_rt']:.0f} ms)",
    )
    print(
        "2. Yes反应时 > No反应时: ",
        f"{'✓' if criteria['yes_rt_greater'] else '✗'} (Yes: {your_data['yes_rt']:.0f} ms, No: {your_data['no_rt']:.0f} ms)",
    )
    print(
        "3. Yes比例合理 (30-80%): ",
        f"{'✓' if criteria['yes_proportion_reasonable'] else '✗'} ({your_data['yes_proportion']:.1%})",
    )
    if criteria["yes_intensity_greater"] is not None:
        print(
            "4. Yes词强度 > No词强度: ",
            f"{'✓' if criteria['yes_intensity_greater'] else '✗'}",
        )

    # 总结评估
    pass_count = sum(
        [1 for v in criteria.values() if v is True or (v is not None and v)]
    )
    total_criteria = len([v for v in criteria.values() if v is not None])

    print(f"\n有效性评估结果: {pass_count}/{total_criteria} 项通过")

    if pass_count == total_criteria:
        print(" 实验数据表现出优秀的有效性，完全符合自我参照编码任务的预期模式。")
    elif pass_count >= total_criteria - 1:
        print(" 实验数据表现出良好的有效性，基本符合预期模式。")
    elif pass_count >= total_criteria - 2:
        print("️  实验数据基本有效，但部分指标偏离预期，需在讨论中说明。")
    else:
        print(" 实验数据有效性不足，建议检查实验程序或数据处理。")

    # 11. 创建可视化
    print("\n️  正在生成可视化图表...")
    create_visualizations(df_analysis, your_data, rogers_ref)

    # 12. 保存汇总结果
    summary_df = create_summary_dataframe(
        df_analysis, your_data, rogers_ref, yes_intensity, no_intensity, overall_corr
    )

    filename = "self_reference_analysis_summary.csv"

    summary_df.write_csv(filename)
    print(f"\n 结果已保存到: {filename}")

    return summary_df


def create_visualizations(df, your_data, rogers_ref):
    """创建可视化图表"""

    # 转换为pandas用于Plotly（如果数据量不大）
    if df.height < 10000:  # 安全阈值
        df_pd = df.to_pandas()
    else:
        print("️  数据量过大，部分可视化可能被跳过")
        df_pd = df.head(1000).to_pandas()

    # 1. 反应时分布图
    fig1 = px.histogram(
        df_pd,
        x="rt_ms",
        color="response",
        nbins=30,
        title="反应时分布",
        labels={"rt_ms": "反应时 (ms)", "count": "频数"},
        opacity=0.7,
        barmode="overlay",
    )
    fig1.update_layout(
        xaxis_range=[0, df["rt_ms"].max() * 1.1], template="plotly_white"
    )
    fig1.show()

    # 2. 反应时与强度散点图
    fig2 = px.scatter(
        df_pd,
        x="rt_ms",
        y="intensity",
        color="response",
        title="反应时与强度关系",
        labels={"rt_ms": "反应时 (ms)", "intensity": "强度评分 (0-10)"},
        hover_data=["stim_word"],
        trendline="ols",
    )
    fig2.update_layout(template="plotly_white")
    fig2.show()

    # 3. 与文献比较的条形图
    comparison_data = {
        "指标": ["平均反应时", "Yes反应时", "No反应时", "Yes比例"],
        "你的实验": [
            your_data["mean_rt"],
            your_data["yes_rt"],
            your_data["no_rt"],
            your_data["yes_proportion"] * 100,  # 转换为百分比
        ],
        "Rogers(1977)": [
            rogers_ref["mean_rt"],
            rogers_ref["yes_rt"],
            rogers_ref["no_rt"],
            rogers_ref["yes_proportion"] * 100,
        ],
    }

    fig3 = go.Figure()

    # 添加你的数据
    fig3.add_trace(
        go.Bar(
            name="你的实验",
            x=comparison_data["指标"],
            y=comparison_data["你的实验"],
            marker_color="indianred",
            text=[
                f"{y:.0f}" if i < 3 else f"{y:.1f}%"
                for i, y in enumerate(comparison_data["你的实验"])
            ],
            textposition="auto",
        )
    )

    # 添加文献数据
    fig3.add_trace(
        go.Bar(
            name="Rogers(1977)",
            x=comparison_data["指标"],
            y=comparison_data["Rogers(1977)"],
            marker_color="lightseagreen",
            text=[
                f"{y:.0f}" if i < 3 else f"{y:.1f}%"
                for i, y in enumerate(comparison_data["Rogers(1977)"])
            ],
            textposition="auto",
        )
    )

    fig3.update_layout(
        title="与经典研究比较",
        xaxis_title="指标",
        yaxis_title="数值",
        barmode="group",
        template="plotly_white",
        yaxis=dict(title="反应时(ms) / 比例(%)", tickformat=",d"),
    )
    fig3.show()

    # 4. 强度评分分布
    if "stim_type" in df.columns:
        fig4 = px.box(
            df_pd,
            x="response",
            y="intensity",
            color="stim_type",
            title="强度评分分布",
            labels={"response": "反应类型", "intensity": "强度评分 (0-10)"},
            points="all",
        )
        fig4.update_layout(template="plotly_white")
        fig4.show()

    # 5. 反应时序列图
    if "trial_index" in df.columns:
        df_seq = df.with_columns(pl.Series("trial_order", range(df.height)))

        fig5 = px.line(
            df_seq.to_pandas(),
            x="trial_order",
            y="rt_ms",
            color="response",
            title="反应时序列变化",
            labels={"trial_order": "试次顺序", "rt_ms": "反应时 (ms)"},
            hover_data=["stim_word", "intensity"],
        )
        fig5.update_layout(template="plotly_white")

        # 添加移动平均线
        for resp in ["yes", "no"]:
            subset = df_seq.filter(pl.col("response") == resp)
            if subset.height > 5:
                window = min(10, subset.height // 5)
                moving_avg = subset["rt_ms"].rolling_mean(
                    window_size=window, min_periods=1
                )

                fig5.add_trace(
                    go.Scatter(
                        x=subset["trial_order"].to_numpy(),
                        y=moving_avg.to_numpy(),
                        name=f"{resp} 移动平均(window={window})",
                        line=dict(dash="dash", width=2),
                        opacity=0.7,
                    )
                )

        fig5.show()


def create_summary_dataframe(
    df, your_data, rogers_ref, yes_intensity, no_intensity, overall_corr
):
    """创建汇总数据框"""

    summary_data = {
        "指标": [
            "总试次数",
            "Yes反应数",
            "Yes比例",
            "平均反应时(ms)",
            "反应时标准差(ms)",
            "反应时中位数(ms)",
            "Yes平均反应时(ms)",
            "No平均反应时(ms)",
            "整体强度评分",
            "Yes平均强度",
            "No平均强度",
            "RT-强度相关性",
            "文献平均RT(ms)",
            "文献Yes比例",
            "RT差异(ms)",
            "Yes比例差异",
        ],
        "数值": [
            df.height,
            df.filter(pl.col("response") == "yes").height,
            your_data["yes_proportion"],
            your_data["mean_rt"],
            df["rt_ms"].std(),
            df["rt_ms"].median(),
            your_data["yes_rt"],
            your_data["no_rt"],
            df["intensity"].mean(),
            np.mean(yes_intensity) if len(yes_intensity) > 0 else None,
            np.mean(no_intensity) if len(no_intensity) > 0 else None,
            overall_corr["correlation"][0],
            rogers_ref["mean_rt"],
            rogers_ref["yes_proportion"],
            your_data["mean_rt"] - rogers_ref["mean_rt"],
            your_data["yes_proportion"] - rogers_ref["yes_proportion"],
        ],
        "单位": [
            "次",
            "次",
            "百分比",
            "ms",
            "ms",
            "ms",
            "ms",
            "ms",
            "分",
            "分",
            "分",
            "相关系数",
            "ms",
            "百分比",
            "ms",
            "百分比",
        ],
    }

    return pl.DataFrame(summary_data, strict=False)


# 主函数
def main():
    """主函数"""

    print("自我参照编码任务数据分析系统")
    print("=" * 50)

    file_path = Path(input("请输入SRET数据文件路径:\n").strip("'").strip()).resolve()
    df = pl.read_csv(file_path)

    target_blocks = ["Encoding"]
    block_col = "phase"

    # 分析数据
    analyze_self_reference_data(
        df,
        target_blocks,
        block_col,
    )


if __name__ == "__main__":
    main()
