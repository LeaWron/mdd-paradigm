import json
import warnings
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from omegaconf import DictConfig, OmegaConf
from pyxdf import load_xdf
from scipy.signal import detrend, find_peaks

warnings.filterwarnings("ignore")

# 设置matplotlib支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 尝试导入Plotly用于交互式可视化
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly不可用，将使用静态图表")


def analyze_xdf_data(xdf_file_path: Path, output_dir: Path):
    """
    分析XDF文件中的多模态数据并生成分析结果

    Parameters:
    -----------
    xdf_file_path : Path
        XDF文件路径
    output_dir : Path
        输出目录
    """

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始分析XDF文件: {xdf_file_path}")

    # 加载XDF文件
    streams, header = load_xdf(str(xdf_file_path))

    # 解析各个流
    eeg_stream = None
    marker_streams = []
    nirs_stream = None

    for stream in streams:
        stream_type = stream["info"]["type"][0]
        if stream_type == "EEG":
            eeg_stream = stream
        elif stream_type == "Markers":
            marker_streams.append(stream)
        elif stream_type == "NIRS":
            nirs_stream = stream
            # 检查fNIRS流的数据状态
            if nirs_stream["time_series"] is not None:
                data_points = len(nirs_stream["time_series"])
                print(f"找到fNIRS流，包含 {data_points} 个数据点")
            else:
                print("找到fNIRS流，但数据为空")

    # 分析结果字典
    analysis_results = {}

    # 1. 分析EEG数据（包含ECG, EOG, GSR）
    if eeg_stream:
        print("分析EEG数据...")
        eeg_results = analyze_eeg_data(eeg_stream, marker_streams, output_dir)
        analysis_results["eeg"] = eeg_results

    # 2. 分析fNIRS数据
    if nirs_stream:
        print("分析fNIRS数据...")
        nirs_results = analyze_nirs_data(nirs_stream, marker_streams, output_dir)
        analysis_results["fnirs"] = nirs_results
    else:
        print("未找到fNIRS流")
        analysis_results["fnirs"] = {"error": "未找到fNIRS流"}

    # 3. 保存总体分析结果
    save_overall_results(analysis_results, output_dir)

    print(f"分析完成！结果保存在: {output_dir}")


def setup_electrode_positions(raw):
    """设置电极位置 - 改进版本"""
    try:
        montage = mne.channels.make_standard_montage("standard_1020")

        # 创建更精确的通道名称映射
        ch_mapping = {}
        standard_names = [
            "Fp1",
            "Fp2",
            "Fz",
            "F3",
            "F4",
            "F7",
            "F8",
            "F9",
            "F10",
            "Cz",
            "C3",
            "C4",
            "T7",
            "T8",
            "Pz",
            "P3",
            "P4",
            "P7",
            "P8",
            "P9",
            "P10",
            "Oz",
            "O1",
            "O2",
            "FC1",
            "FC2",
            "FC5",
            "FC6",
            "CP1",
            "CP2",
            "CP5",
            "CP6",
        ]

        # 改进的映射逻辑
        mapping_found = 0
        for orig_name in raw.ch_names:
            orig_upper = orig_name.upper()

            # 尝试精确匹配
            matched = False
            for std_name in standard_names:
                if orig_upper == std_name.upper():
                    ch_mapping[orig_name] = std_name
                    mapping_found += 1
                    matched = True
                    break

            # 如果没有精确匹配，尝试模糊匹配
            if not matched:
                # 处理常见的命名变体
                if orig_upper in ["FP1", "Fp1"]:
                    ch_mapping[orig_name] = "Fp1"
                    mapping_found += 1
                elif orig_upper in ["FP2", "Fp2"]:
                    ch_mapping[orig_name] = "Fp2"
                    mapping_found += 1
                elif orig_upper == "OZ":
                    ch_mapping[orig_name] = "Oz"
                    mapping_found += 1
                elif orig_upper == "CZ":
                    ch_mapping[orig_name] = "Cz"
                    mapping_found += 1
                elif orig_upper == "PZ":
                    ch_mapping[orig_name] = "Pz"
                    mapping_found += 1
                elif orig_upper == "FZ":
                    ch_mapping[orig_name] = "Fz"
                    mapping_found += 1
                else:
                    # 保持原名称
                    ch_mapping[orig_name] = orig_name

        print(f"成功映射 {mapping_found}/{len(raw.ch_names)} 个通道到标准名称")

        # 应用映射
        raw.rename_channels(ch_mapping)

        # 设置蒙太奇
        raw.set_montage(montage, on_missing="warn")  # 使用warn来查看哪些通道没匹配

        return True

    except Exception as e:
        print(f"设置电极位置失败: {e}")
        return False


def analyze_eeg_data(eeg_stream, marker_streams, output_dir):
    """分析EEG数据，包括ECG, EOG, GSR"""

    # 提取数据和时间戳
    eeg_data = eeg_stream["time_series"].T
    timestamps = eeg_stream["time_stamps"]
    info = eeg_stream["info"]

    # 获取通道信息
    channels_info = info["desc"][0]["channels"][0]["channel"]
    channel_names = [ch["label"][0] for ch in channels_info]

    # 修正：通道类型必须小写，并正确映射
    channel_types = []
    for ch in channels_info:
        ch_type = ch["type"][0].lower()
        if ch_type == "eeg":
            channel_types.append("eeg")
        elif ch_type == "aux":
            label_lower = ch["label"][0].lower()
            if "ecg" in label_lower or "ekg" in label_lower:
                channel_types.append("ecg")
            elif "eog" in label_lower:
                channel_types.append("eog")
            elif "gsr" in label_lower or "eda" in label_lower:
                channel_types.append("gsr")
            else:
                channel_types.append("misc")
        else:
            channel_types.append(ch_type)

    # 创建MNE Info对象
    sfreq = float(info["nominal_srate"][0])

    eeg_info = mne.create_info(
        ch_names=channel_names, sfreq=sfreq, ch_types=channel_types
    )

    # 创建Raw对象
    raw = mne.io.RawArray(eeg_data, eeg_info)

    # 设置电极位置
    setup_success = setup_electrode_positions(raw)

    # 关键修复：在重命名后重新获取EEG通道列表
    # 使用当前raw对象中的通道名称和类型
    current_eeg_chs = [
        ch for ch in raw.ch_names if raw.get_channel_types(picks=[ch])[0] == "eeg"
    ]
    current_aux_chs = [
        ch
        for ch in raw.ch_names
        if raw.get_channel_types(picks=[ch])[0] in ["ecg", "eog", "gsr", "misc"]
    ]

    # 根据通道标签识别生理信号
    ecg_chs = [
        ch for ch in current_aux_chs if "ecg" in ch.lower() or "ekg" in ch.lower()
    ]
    eog_chs = [ch for ch in current_aux_chs if "eog" in ch.lower()]
    gsr_chs = [
        ch for ch in current_aux_chs if "gsr" in ch.lower() or "eda" in ch.lower()
    ]

    # 分析结果字典
    results = {
        "basic_info": {
            "sampling_rate": sfreq,
            "duration": len(timestamps) / sfreq,
            "eeg_channels": current_eeg_chs,
            "aux_channels": current_aux_chs,
            "ecg_channels": ecg_chs,
            "eog_channels": eog_chs,
            "gsr_channels": gsr_chs,
            "electrode_setup_success": setup_success,
        }
    }

    # 1. EEG分析 - 使用更新后的通道列表
    if current_eeg_chs:
        print(f"进行EEG分析，通道: {current_eeg_chs}")
        eeg_results = analyze_eeg_signals(
            raw.copy().pick(current_eeg_chs), output_dir / "eeg"
        )
        results["eeg_analysis"] = eeg_results
    else:
        print("警告: 没有找到EEG通道")
        results["eeg_analysis"] = {}

    # 2. ECG分析
    if ecg_chs:
        print(f"进行ECG分析，通道: {ecg_chs}")
        ecg_results = analyze_ecg_signals(raw.copy().pick(ecg_chs), output_dir / "ecg")
        results["ecg_analysis"] = ecg_results
    else:
        print("未找到ECG通道")
        results["ecg_analysis"] = {"error": "未找到ECG通道"}

    # 3. EOG分析
    if eog_chs:
        print(f"进行EOG分析，通道: {eog_chs}")
        eog_results = analyze_eog_signals(raw.copy().pick(eog_chs), output_dir / "eog")
        results["eog_analysis"] = eog_results
    else:
        print("未找到EOG通道")
        results["eog_analysis"] = {"error": "未找到EOG通道"}

    # 4. GSR分析
    if gsr_chs:
        print(f"进行GSR分析，通道: {gsr_chs}")
        gsr_results = analyze_gsr_signals(raw.copy().pick(gsr_chs), output_dir / "gsr")
        results["gsr_analysis"] = gsr_results
    else:
        print("未找到GSR通道")
        results["gsr_analysis"] = {"error": "未找到GSR通道"}

    # 5. 事件相关的分析
    events = extract_events_from_markers(marker_streams, timestamps, sfreq)
    if events is not None and len(events) > 0:
        print(f"进行事件相关分析，找到 {len(events)} 个事件")
        event_results = analyze_event_related_data(raw, events, output_dir / "events")
        results["event_analysis"] = event_results
    else:
        print("未找到事件标记")
        results["event_analysis"] = {"error": "未找到事件标记"}

    # 6. 保存原始数据曲线（包括交互式HTML）
    save_raw_data_plots(raw, current_eeg_chs, ecg_chs, eog_chs, gsr_chs, output_dir)

    return results


def save_raw_data_plots(raw, eeg_chs, ecg_chs, eog_chs, gsr_chs, output_dir):
    """保存所有原始数据曲线图（包括交互式HTML）"""

    # 保存EEG原始数据曲线
    if eeg_chs:
        try:
            eeg_raw = raw.copy().pick(eeg_chs)
            # 限制显示的通道数量，避免图像过于拥挤
            n_channels_to_plot = min(len(eeg_chs), 20)
            eeg_plot_chs = eeg_chs[:n_channels_to_plot]
            eeg_raw_plot = raw.copy().pick(eeg_plot_chs)

            # 绘制前60秒的EEG数据
            duration = min(60, eeg_raw.times[-1])
            eeg_raw_plot_cropped = eeg_raw_plot.copy().crop(tmax=duration)

            # 使用MNE的plot方法生成静态图
            fig = eeg_raw_plot_cropped.plot(
                show=False,
                title=f"EEG原始信号 (前{duration}秒，显示{n_channels_to_plot}个通道)",
                scalings="auto",
            )
            fig.set_size_inches(15, 10)
            plt.savefig(
                output_dir / "eeg" / "eeg_raw_signals.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig)
            print("成功保存EEG原始信号静态图")

            # 生成交互式HTML图
            if PLOTLY_AVAILABLE:
                create_interactive_raw_plot(
                    eeg_raw_plot,
                    output_dir / "eeg" / "eeg_raw_signals_interactive.html",
                    "EEG原始信号",
                    scalings="auto",
                )
                print("成功保存EEG原始信号交互式图")

        except Exception as e:
            print(f"保存EEG原始信号图失败: {e}")


def create_interactive_raw_plot(
    raw, output_path, title, scalings="auto", max_duration=300
):
    """
    创建交互式原始信号图

    Parameters:
    -----------
    raw : mne.io.Raw
        MNE原始数据对象
    output_path : Path
        输出HTML文件路径
    title : str
        图表标题
    scalings : dict or 'auto'
        通道缩放因子
    max_duration : float
        最大显示时长（秒）
    """

    # 限制显示时长
    duration = min(max_duration, raw.times[-1])
    raw_cropped = raw.copy().crop(tmax=duration)

    # 获取数据和时间
    data, times = raw_cropped[:, :]
    ch_names = raw_cropped.ch_names

    # 计算合适的缩放因子
    if scalings == "auto":
        # 自动计算缩放因子
        scalings = {}
        for i, ch_name in enumerate(ch_names):
            ch_data = data[i]
            data_range = np.max(ch_data) - np.min(ch_data)
            if data_range > 0:
                # 使每个通道的幅度范围大致为1
                scalings[ch_name] = 1.0 / data_range
            else:
                scalings[ch_name] = 1.0

    # 创建子图
    fig = make_subplots(
        rows=len(ch_names),
        cols=1,
        subplot_titles=ch_names,
        vertical_spacing=0.02,
        shared_xaxes=True,
    )

    # 为每个通道添加轨迹
    for i, ch_name in enumerate(ch_names):
        ch_data = data[i]

        # 应用缩放
        if ch_name in scalings:
            scaled_data = ch_data * scalings[ch_name]
        else:
            scaled_data = ch_data

        # 添加偏移以便区分通道
        offset = i * 2
        y_data = scaled_data + offset

        fig.add_trace(
            go.Scatter(
                x=times,
                y=y_data,
                mode="lines",
                name=ch_name,
                line=dict(width=1),
                hovertemplate=(
                    f"通道: {ch_name}<br>"
                    + "时间: %{x:.2f}秒<br>"
                    + "原始值: %{customdata:.4f}<br>"
                    + "缩放值: %{y:.4f}<extra></extra>"
                ),
                customdata=ch_data,  # 保存原始数据用于悬停显示
            ),
            row=i + 1,
            col=1,
        )

        # 设置y轴标签
        fig.update_yaxes(
            title_text=ch_name,
            row=i + 1,
            col=1,
            tickvals=[offset],
            ticktext=[ch_name],
            showgrid=True,
        )

    # 更新布局
    fig.update_layout(
        height=100 * len(ch_names),
        title_text=f"{title} (前{duration}秒)",
        showlegend=False,
        hovermode="x unified",
        xaxis_title="时间 (秒)",
    )

    # 添加分辨率调节按钮
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list(
                    [
                        dict(
                            args=[{"visible": [True] * len(ch_names)}],
                            label="显示所有通道",
                            method="restyle",
                        ),
                        dict(
                            args=[{"visible": [i < 10 for i in range(len(ch_names))]}],
                            label="显示前10个通道",
                            method="restyle",
                        ),
                    ]
                ),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
        ]
    )

    # 保存为HTML文件
    fig.write_html(str(output_path), config={"responsive": True})


def analyze_eeg_signals(raw, output_dir):
    """分析EEG信号 - 改进版本"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 基本预处理
    raw.filter(1, 40)

    # 计算PSD
    spectrum = raw.compute_psd(method="welch", fmin=1, fmax=40, n_fft=1024)
    psds, freqs = spectrum.get_data(return_freqs=True)

    # 保存PSD图
    fig = spectrum.plot(show=False)
    fig.savefig(output_dir / "eeg_psd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 计算各频段功率
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 40),
    }

    band_power = {}
    for band, (fmin, fmax) in bands.items():
        band_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
        band_power[band] = np.mean(psds[:, band_idx], axis=1)

    results["band_power"] = band_power
    results["psd_freqs"] = freqs.tolist()
    results["psd_values"] = psds.tolist()

    # ICA分析 - 改进版本：不限制组件上限
    try:
        # 根据实际通道数量动态调整ICA组件数量
        n_channels = len(raw.ch_names)

        # 不限制ICA组件上限，只设置最小值为2
        n_components = max(n_channels, 2)  # 使用所有通道作为ICA组件

        print(f"EEG通道数: {n_channels}, 使用ICA组件数: {n_components}")

        # 创建ICA对象
        ica = ICA(n_components=n_components, random_state=97, max_iter=1000)
        ica.fit(raw)

        # 保存ICA相关信息
        results["ica_components"] = (
            ica.get_components().tolist() if hasattr(ica, "get_components") else []
        )
        results["ica_mixing_matrix"] = (
            ica.mixing_matrix_.tolist() if hasattr(ica, "mixing_matrix_") else []
        )
        results["ica_n_components"] = n_components
        results["ica_channel_names"] = raw.ch_names  # 保存通道名称

        # 1. 绘制ICA组件拓扑图（如果可能）
        try:
            # 检查是否有电极位置
            if hasattr(raw.info, "dig") and raw.info["dig"] is not None:
                fig = ica.plot_components(show=False)
                # 改进图片标题，包含通道信息
                fig.suptitle(
                    f"ICA组件 - {n_components}个组件来自{n_channels}个通道", fontsize=12
                )
                fig.savefig(
                    output_dir / "eeg_ica_components.png", dpi=300, bbox_inches="tight"
                )
                plt.close(fig)
                print("成功保存ICA组件拓扑图")
            else:
                print("警告: 没有电极位置信息，无法绘制ICA组件拓扑图")
                # 创建一个简单的通道列表图作为替代
                plt.figure(figsize=(12, 8))
                plt.text(
                    0.1,
                    0.5,
                    f"ICA分析使用通道 ({n_channels} channels):\n"
                    + "\n".join(raw.ch_names),
                    fontfamily="monospace",
                    fontsize=10,
                    va="center",
                )
                plt.title(f"ICA通道 - {n_components}个组件来自{n_channels}个通道")
                plt.axis("off")
                plt.savefig(
                    output_dir / "eeg_ica_channels.png", dpi=300, bbox_inches="tight"
                )
                plt.close()
        except Exception as e:
            print(f"绘制ICA组件图失败: {e}")

        # 2. 生成交互式ICA源信号图
        try:
            # 获取ICA源信号数据
            ica_sources = ica.get_sources(raw)
            sources_data = ica_sources.get_data()
            times = ica_sources.times

            # 创建交互式Plotly图表
            if PLOTLY_AVAILABLE:
                create_interactive_ica_plot(
                    sources_data, times, n_components, raw.ch_names, output_dir
                )
                print("成功生成交互式ICA源信号图")
            else:
                print("Plotly不可用，生成静态ICA图")
                create_static_ica_images(
                    sources_data, times, n_components, output_dir, raw.info["sfreq"]
                )

        except Exception as e:
            print(f"生成ICA图失败: {e}")
            # 回退到静态图
            try:
                display_duration = min(60, raw.times[-1])
                raw_display = raw.copy().crop(tmax=display_duration)
                fig = ica.plot_sources(raw_display, show=False)
                fig.suptitle(
                    f"ICA源信号 - {n_components}个组件\n显示前{display_duration}秒",
                    fontsize=10,
                )
                fig.savefig(
                    output_dir / "eeg_ica_sources.png", dpi=300, bbox_inches="tight"
                )
                plt.close(fig)
                print("成功保存静态ICA源信号图")
            except Exception as e2:
                print(f"保存静态ICA源信号图也失败: {e2}")

        # 3. 保存ICA详细信息
        try:
            with open(output_dir / "ica_details.txt", "w", encoding="utf-8") as f:
                f.write("ICA分析详细信息\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"数据总时长: {raw.times[-1]:.2f} 秒\n")
                f.write(f"EEG通道数量: {n_channels}\n")
                f.write(f"ICA组件数量: {n_components}\n")
                f.write(f"采样率: {raw.info['sfreq']} Hz\n\n")

                f.write("使用的通道:\n")
                for i, ch_name in enumerate(raw.ch_names):
                    f.write(f"  {i + 1:2d}. {ch_name}\n")

                if hasattr(ica, "explained_variance_ratio_"):
                    f.write(
                        f"\nICA解释的方差: {ica.explained_variance_ratio_.sum():.3f}\n"
                    )

            print("成功保存ICA详细信息")

        except Exception as e:
            print(f"保存ICA详细信息失败: {e}")

        results["ica_success"] = True

    except Exception as e:
        print(f"ICA分析失败: {e}")
        results["ica_success"] = False
        results["ica_components"] = []
        results["ica_mixing_matrix"] = []

    return results


def create_interactive_ica_plot(
    sources_data, times, n_components, channel_names, output_dir
):
    """创建交互式ICA源信号图"""

    # 不限制显示的组件数量，显示所有组件
    max_display_components = n_components

    # 创建子图
    fig = make_subplots(
        rows=max_display_components,
        cols=1,
        subplot_titles=[f"ICA组件 {i + 1}" for i in range(max_display_components)],
        vertical_spacing=0.02,
    )

    # 为每个组件添加轨迹
    for i in range(max_display_components):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=sources_data[i],
                mode="lines",
                name=f"ICA {i + 1}",
                line=dict(width=1),
                hovertemplate="时间: %{x:.2f}秒<br>幅度: %{y:.4f}<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )

    # 更新布局
    fig.update_layout(
        height=200 * max_display_components,
        title_text=f"ICA源信号 - 显示所有{max_display_components}个组件",
        showlegend=False,
        hovermode="x unified",
    )

    # 更新x轴和y轴
    for i in range(max_display_components):
        fig.update_xaxes(title_text="时间 (秒)", row=i + 1, col=1)
        fig.update_yaxes(title_text=f"组件 {i + 1}", row=i + 1, col=1)

    # 保存为HTML文件
    html_file = output_dir / "eeg_ica_sources_interactive.html"
    fig.write_html(str(html_file), config={"responsive": True})

    # 也保存为静态图片（前60秒）
    static_times = times[times <= 60] if len(times) > 0 else times
    if len(static_times) > 0:
        static_indices = np.where(times <= 60)[0]
        if len(static_indices) > 0:
            fig_static = make_subplots(
                rows=min(10, max_display_components),
                cols=1,
                subplot_titles=[
                    f"ICA组件 {i + 1}" for i in range(min(10, max_display_components))
                ],
                vertical_spacing=0.03,
            )

            for i in range(min(10, max_display_components)):
                fig_static.add_trace(
                    go.Scatter(
                        x=static_times,
                        y=sources_data[i, static_indices],
                        mode="lines",
                        name=f"ICA {i + 1}",
                        line=dict(width=1),
                    ),
                    row=i + 1,
                    col=1,
                )

            fig_static.update_layout(
                height=1500,
                title_text="ICA源信号 - 前60秒 (前10个组件)",
                showlegend=False,
            )

            for i in range(min(10, max_display_components)):
                fig_static.update_xaxes(title_text="时间 (秒)", row=i + 1, col=1)
                fig_static.update_yaxes(title_text=f"组件 {i + 1}", row=i + 1, col=1)

            fig_static.write_image(
                str(output_dir / "eeg_ica_sources_static.png"), width=1200, height=1500
            )


def create_static_ica_images(sources_data, times, n_components, output_dir, sfreq):
    """创建多张静态ICA图片"""

    # 每张图片显示的组件数量
    components_per_image = 6
    n_images = (n_components + components_per_image - 1) // components_per_image

    for img_idx in range(n_images):
        start_comp = img_idx * components_per_image
        end_comp = min((img_idx + 1) * components_per_image, n_components)

        fig, axes = plt.subplots(components_per_image, 1, figsize=(15, 12))
        if components_per_image == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            comp_idx = start_comp + i
            if comp_idx < n_components:
                # 显示前60秒
                display_samples = min(len(times), int(60 * sfreq))
                ax.plot(
                    times[:display_samples],
                    sources_data[comp_idx, :display_samples],
                    linewidth=0.8,
                )
                ax.set_title(f"ICA组件 {comp_idx + 1}", fontsize=10)
                ax.set_ylabel("幅度", fontsize=8)
                ax.grid(True, alpha=0.3)

                if i == components_per_image - 1:
                    ax.set_xlabel("时间 (秒)", fontsize=8)
                else:
                    ax.set_xticklabels([])

        plt.suptitle(f"ICA源信号 - 组件 {start_comp + 1} 到 {end_comp}", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"ica_sources_page_{img_idx + 1}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def analyze_ecg_signals(raw, output_dir):
    """分析ECG信号 - 使用替代方法"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 预处理
    raw.filter(0.5, 40)

    ecg_data = raw.get_data()[0]  # 假设只有一个ECG通道

    # 使用简单的峰值检测
    peaks, _ = find_peaks(
        ecg_data, height=np.std(ecg_data) * 2, distance=int(raw.info["sfreq"] * 0.6)
    )  # 最小间隔0.6秒
    r_peaks = peaks

    # 计算心率变异性
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / raw.info["sfreq"] * 1000  # 转换为毫秒
        heart_rate = 60 / (rr_intervals / 1000)  # 转换为BPM

        results["r_peaks"] = r_peaks.tolist()
        results["rr_intervals"] = rr_intervals.tolist()
        results["heart_rate"] = heart_rate.tolist()
        results["mean_heart_rate"] = float(np.mean(heart_rate))
        results["hrv_rmssd"] = float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2)))
    else:
        results["r_peaks"] = r_peaks.tolist()
        results["rr_intervals"] = []
        results["heart_rate"] = []
        results["mean_heart_rate"] = 0
        results["hrv_rmssd"] = 0

    # 绘制ECG信号
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(len(ecg_data)) / raw.info["sfreq"]
    plt.plot(time_axis, ecg_data, label="ECG", linewidth=1)
    plt.plot(time_axis[r_peaks], ecg_data[r_peaks], "ro", label="R峰", markersize=4)
    plt.xlabel("时间 (秒)")
    plt.ylabel("幅度")
    plt.title(f"ECG信号与R峰检测\n平均心率: {results['mean_heart_rate']:.1f} BPM")
    plt.legend()
    plt.savefig(output_dir / "ecg_rpeaks.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 保存ECG原始信号
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, ecg_data, linewidth=1)
    plt.xlabel("时间 (秒)")
    plt.ylabel("幅度")
    plt.title("ECG原始信号")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "ecg_raw_signal.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 生成交互式ECG图
    if PLOTLY_AVAILABLE:
        create_interactive_single_channel_plot(
            ecg_data,
            time_axis,
            "ECG原始信号",
            output_dir / "ecg_raw_signal_interactive.html",
            r_peaks=r_peaks,
        )
        print("成功保存ECG交互式信号图")

    return results


def analyze_eog_signals(raw, output_dir):
    """分析EOG信号"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 预处理
    raw.filter(0.1, 15)

    eog_data = raw.get_data()
    time_axis = np.arange(eog_data.shape[1]) / raw.info["sfreq"]

    # 检测眨眼事件
    threshold = np.std(eog_data) * 3
    blink_events = []

    for i in range(eog_data.shape[1]):
        if np.any(np.abs(eog_data[:, i]) > threshold):
            blink_events.append(i)

    blink_rate = (
        len(blink_events) / (len(eog_data[0]) / raw.info["sfreq"]) * 60
    )  # 眨眼/分钟

    results["blink_events"] = blink_events
    results["blink_rate"] = blink_rate
    results["eog_amplitude"] = float(np.std(eog_data))

    # 绘制EOG信号
    plt.figure(figsize=(12, 6))
    for i, channel in enumerate(raw.ch_names):
        plt.plot(time_axis, eog_data[i], label=channel, linewidth=1)
    plt.xlabel("时间 (秒)")
    plt.ylabel("幅度")
    plt.title(f"EOG信号\n眨眼率: {blink_rate:.1f} 次/分钟")
    plt.legend()
    plt.savefig(output_dir / "eog_signals.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 保存EOG原始信号
    plt.figure(figsize=(12, 6))
    for i, channel in enumerate(raw.ch_names):
        plt.plot(time_axis, eog_data[i], label=channel, linewidth=1)
    plt.xlabel("时间 (秒)")
    plt.ylabel("幅度")
    plt.title("EOG原始信号")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "eog_raw_signals.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 生成交互式EOG图
    if PLOTLY_AVAILABLE:
        create_interactive_multi_channel_plot(
            eog_data,
            time_axis,
            raw.ch_names,
            "EOG原始信号",
            output_dir / "eog_raw_signals_interactive.html",
        )
        print("成功保存EOG交互式信号图")

    return results


def analyze_gsr_signals(raw, output_dir):
    """分析GSR信号"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    gsr_data = raw.get_data()[0]  # 假设只有一个GSR通道
    time_axis = np.arange(len(gsr_data)) / raw.info["sfreq"]

    # 预处理
    gsr_filtered = detrend(gsr_data)

    # 提取特征
    results["mean_gsr"] = float(np.mean(gsr_filtered))
    results["std_gsr"] = float(np.std(gsr_filtered))
    results["max_gsr"] = float(np.max(gsr_filtered))
    results["min_gsr"] = float(np.min(gsr_filtered))

    # 检测SCR (Skin Conductance Response)
    scr_peaks, _ = find_peaks(gsr_filtered, height=np.std(gsr_filtered))
    results["scr_peaks"] = scr_peaks.tolist()
    results["scr_count"] = len(scr_peaks)

    # 绘制GSR信号
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, gsr_data, label="原始GSR", alpha=0.7, linewidth=1)
    plt.plot(time_axis, gsr_filtered, label="去趋势GSR", linewidth=1)
    plt.plot(
        time_axis[scr_peaks],
        gsr_filtered[scr_peaks],
        "ro",
        label="SCR峰值",
        markersize=4,
    )
    plt.xlabel("时间 (秒)")
    plt.ylabel("幅度")
    plt.title(f"GSR信号分析\nSCR事件数: {len(scr_peaks)}")
    plt.legend()
    plt.savefig(output_dir / "gsr_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 保存GSR原始信号
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, gsr_data, linewidth=1)
    plt.xlabel("时间 (秒)")
    plt.ylabel("幅度")
    plt.title("GSR原始信号")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "gsr_raw_signal.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 生成交互式GSR图
    if PLOTLY_AVAILABLE:
        create_interactive_single_channel_plot(
            gsr_data,
            time_axis,
            "GSR原始信号",
            output_dir / "gsr_raw_signal_interactive.html",
            peaks=scr_peaks,
            peak_label="SCR峰值",
        )
        print("成功保存GSR交互式信号图")

    return results


def create_interactive_single_channel_plot(
    data, times, title, output_path, peaks=None, peak_label="峰值", max_duration=300
):
    """创建单通道交互式信号图"""

    # 限制显示时长
    duration = min(max_duration, times[-1])
    end_idx = np.where(times <= duration)[0][-1] + 1

    times_cropped = times[:end_idx]
    data_cropped = data[:end_idx]

    fig = go.Figure()

    # 添加主信号
    fig.add_trace(
        go.Scatter(
            x=times_cropped,
            y=data_cropped,
            mode="lines",
            name="信号",
            line=dict(width=1),
            hovertemplate="时间: %{x:.2f}秒<br>幅度: %{y:.4f}<extra></extra>",
        )
    )

    # 添加峰值标记（如果有）
    if peaks is not None:
        peak_times = times[peaks]
        peak_values = data[peaks]
        peak_mask = peak_times <= duration

        if np.any(peak_mask):
            fig.add_trace(
                go.Scatter(
                    x=peak_times[peak_mask],
                    y=peak_values[peak_mask],
                    mode="markers",
                    name=peak_label,
                    marker=dict(size=8, color="red"),
                    hovertemplate=f"{peak_label}<br>时间: %{{x:.2f}}秒<br>幅度: %{{y:.4f}}<extra></extra>",
                )
            )

    # 更新布局
    fig.update_layout(
        height=600,
        title_text=f"{title} (前{duration}秒)",
        xaxis_title="时间 (秒)",
        yaxis_title="幅度",
        hovermode="x unified",
    )

    # 保存为HTML文件
    fig.write_html(str(output_path), config={"responsive": True})


def create_interactive_multi_channel_plot(
    data, times, channel_names, title, output_path, max_duration=300
):
    """创建多通道交互式信号图"""

    # 限制显示时长
    duration = min(max_duration, times[-1])
    end_idx = np.where(times <= duration)[0][-1] + 1

    times_cropped = times[:end_idx]
    data_cropped = data[:, :end_idx]

    # 创建子图
    fig = make_subplots(
        rows=len(channel_names),
        cols=1,
        subplot_titles=channel_names,
        vertical_spacing=0.02,
        shared_xaxes=True,
    )

    # 为每个通道添加轨迹
    for i, ch_name in enumerate(channel_names):
        ch_data = data_cropped[i]

        fig.add_trace(
            go.Scatter(
                x=times_cropped,
                y=ch_data,
                mode="lines",
                name=ch_name,
                line=dict(width=1),
                hovertemplate=f"通道: {ch_name}<br>时间: %{{x:.2f}}秒<br>幅度: %{{y:.4f}}<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )

        # 设置y轴标签
        fig.update_yaxes(title_text=ch_name, row=i + 1, col=1)

    # 更新布局
    fig.update_layout(
        height=200 * len(channel_names),
        title_text=f"{title} (前{duration}秒)",
        showlegend=False,
        hovermode="x unified",
        xaxis_title="时间 (秒)",
    )

    # 保存为HTML文件
    fig.write_html(str(output_path), config={"responsive": True})


def analyze_nirs_data(nirs_stream, marker_streams, output_dir):
    """分析fNIRS数据 - 修复空数据问题"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 检查fNIRS数据是否为空
    if nirs_stream["time_series"] is None or len(nirs_stream["time_series"]) == 0:
        print("警告: fNIRS数据为空，跳过分析")
        results["error"] = "fNIRS数据为空"
        return results

    # 提取数据
    nirs_data = nirs_stream["time_series"].T
    timestamps = nirs_stream["time_stamps"]
    info = nirs_stream["info"]

    # 检查数据维度
    if nirs_data.size == 0:
        print("警告: fNIRS数据为空，跳过分析")
        results["error"] = "fNIRS数据为空"
        return results

    # 获取通道信息
    channels_info = info["desc"][0]["channels"][0]["channel"]
    channel_names = [ch["label"][0] for ch in channels_info]
    channel_types = [ch["type"][0] for ch in channels_info]

    # 检查采样率
    try:
        sfreq = float(info["nominal_srate"][0])
    except Exception as e:
        print(f"警告: 无法获取fNIRS采样率，使用默认值10Hz，错误: {e}")
        sfreq = 10.0

    # 分离不同类型的NIRS通道
    raw_channels = [
        name for name, type_ in zip(channel_names, channel_types) if "nirs_raw" in type_
    ]
    hbo_channels = [
        name for name, type_ in zip(channel_names, channel_types) if "nirs_hbo" in type_
    ]
    hbr_channels = [
        name for name, type_ in zip(channel_names, channel_types) if "nirs_hbr" in type_
    ]

    # 创建NIRS Raw对象
    if hbo_channels:  # 优先使用HBO数据
        try:
            ch_idx = [channel_names.index(ch) for ch in hbo_channels]
            nirs_data_selected = nirs_data[ch_idx]
            ch_types = ["hbo"] * len(hbo_channels)
            ch_names = hbo_channels
        except Exception as e:
            print(f"处理HBO通道时出错: {e}")
            return results
    elif raw_channels:  # 如果没有HBO，使用原始数据
        try:
            ch_idx = [channel_names.index(ch) for ch in raw_channels]
            nirs_data_selected = nirs_data[ch_idx]
            ch_types = ["fnirs_cw_amplitude"] * len(raw_channels)
            ch_names = raw_channels
        except Exception as e:
            print(f"处理原始fNIRS通道时出错: {e}")
            return results
    else:
        print("警告: 未找到可用的NIRS通道")
        results["error"] = "未找到可用的NIRS通道"
        return results

    # 检查数据是否为空
    if nirs_data_selected.size == 0:
        print("警告: 选择的NIRS数据为空")
        results["error"] = "选择的NIRS数据为空"
        return results

    try:
        nirs_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw_nirs = mne.io.RawArray(nirs_data_selected, nirs_info)

        # 基本预处理 - 添加数据检查
        if raw_nirs.n_times > 10:  # 确保有足够的数据点进行滤波
            try:
                raw_nirs.filter(0.01, 0.5)  # 带通滤波
                print("fNIRS数据滤波成功")
            except Exception as e:
                print(f"fNIRS数据滤波失败: {e}")
                # 继续处理，但不进行滤波
        else:
            print("fNIRS数据点太少，跳过滤波")

        results["basic_info"] = {
            "sampling_rate": sfreq,
            "raw_channels": raw_channels,
            "hbo_channels": hbo_channels,
            "hbr_channels": hbr_channels,
            "data_points": raw_nirs.n_times,
        }

        # 计算平均信号
        if raw_nirs.n_times > 0:
            mean_hbo = np.mean(raw_nirs.get_data(), axis=0)
            results["mean_signal"] = mean_hbo.tolist()
            results["signal_variance"] = np.var(raw_nirs.get_data(), axis=1).tolist()
        else:
            results["mean_signal"] = []
            results["signal_variance"] = []

        # 绘制NIRS信号（如果数据足够）
        if raw_nirs.n_times > 1:
            try:
                plt.figure(figsize=(12, 8))
                time_axis = np.arange(raw_nirs.n_times) / sfreq
                data = raw_nirs.get_data()

                # 只绘制前10个通道以避免过于拥挤
                n_plot_channels = min(10, data.shape[0])
                for i in range(n_plot_channels):
                    # 标准化并偏移以便查看
                    if np.std(data[i]) > 0:  # 避免除以零
                        normalized_data = (data[i] - np.mean(data[i])) / np.std(data[i])
                    else:
                        normalized_data = data[i] - np.mean(data[i])
                    plt.plot(
                        time_axis,
                        normalized_data + i * 2,
                        label=raw_nirs.ch_names[i],
                        linewidth=1,
                    )

                plt.xlabel("时间 (秒)")
                plt.ylabel("标准化幅度 (为清晰起见偏移)")
                plt.title(f"fNIRS信号 (显示 {n_plot_channels}/{data.shape[0]} 个通道)")
                plt.legend()
                plt.savefig(
                    output_dir / "fnirs_signals.png", dpi=300, bbox_inches="tight"
                )
                plt.close()
                print("成功保存fNIRS信号图")
            except Exception as e:
                print(f"绘制fNIRS信号图失败: {e}")
        else:
            print("fNIRS数据点不足，跳过绘图")

        # 事件相关分析
        events = extract_events_from_markers(marker_streams, timestamps, sfreq)
        if events is not None and len(events) > 0 and raw_nirs.n_times > 10:
            try:
                event_results = analyze_nirs_events(raw_nirs, events, output_dir)
                results["event_analysis"] = event_results
            except Exception as e:
                print(f"fNIRS事件分析失败: {e}")
                results["event_analysis_error"] = str(e)
        else:
            print("fNIRS事件分析条件不满足")

    except Exception as e:
        print(f"创建fNIRS Raw对象失败: {e}")
        results["error"] = f"创建fNIRS Raw对象失败: {str(e)}"

    return results


def analyze_nirs_events(raw_nirs, events, output_dir):
    """分析fNIRS事件相关数据 - 修复版本"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    try:
        # 检查数据是否足够
        if raw_nirs.n_times < 10:
            results["error"] = "数据点不足"
            return results

        # 创建epochs - 添加preload=True参数和错误处理
        epochs = mne.Epochs(
            raw_nirs,
            events,
            tmin=-2.0,
            tmax=10.0,
            baseline=(-2.0, 0),
            preload=True,
            verbose=False,
        )

        # 检查是否有有效的epochs
        if len(epochs) == 0:
            results["error"] = "没有有效的epochs"
            return results

        # 计算事件相关平均
        evoked = epochs.average()

        results["evoked_data"] = evoked.get_data().tolist()
        results["evoked_times"] = evoked.times.tolist()
        results["n_epochs"] = len(epochs)

        # 尝试绘制结果
        try:
            fig = evoked.plot(show=False)
            fig.savefig(output_dir / "fnirs_erp_plot.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"绘制fNIRS ERP图失败: {e}")

    except Exception as e:
        print(f"fNIRS事件分析失败: {e}")
        results["error"] = str(e)

    return results


def extract_events_from_markers(marker_streams, timestamps, sfreq):
    """从marker流中提取事件"""
    events = []

    for marker_stream in marker_streams:
        if (
            marker_stream["time_series"] is not None
            and len(marker_stream["time_series"]) > 0
        ):
            marker_data = marker_stream["time_series"]
            marker_times = marker_stream["time_stamps"]

            # 将marker时间转换为样本索引
            for i, (marker_time, marker_value) in enumerate(
                zip(marker_times, marker_data)
            ):
                # 找到最接近的时间戳索引
                sample_idx = np.argmin(np.abs(timestamps - marker_time))
                event_id = 1  # 默认事件ID

                # 解析marker值
                if isinstance(marker_value, (list, np.ndarray)):
                    marker_str = str(marker_value[0]) if len(marker_value) > 0 else "1"
                else:
                    marker_str = str(marker_value)

                # 尝试从marker字符串中提取数字作为事件ID
                try:
                    event_id = int("".join(filter(str.isdigit, marker_str)))
                except ValueError:
                    event_id = (
                        hash(marker_str) % 1000 + 1
                    )  # 哈希作为备用方案，确保不为0

                events.append([sample_idx, 0, event_id])

    if events:
        return np.array(events)
    return None


def analyze_event_related_data(raw, events, output_dir):
    """分析事件相关数据"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 创建epochs - 添加preload=True参数
    epochs = mne.Epochs(
        raw, events, tmin=-0.2, tmax=1.0, baseline=(-0.2, 0), preload=True
    )  # 关键修复：添加preload=True

    # 计算ERP
    evoked = epochs.average()

    # 保存ERP图
    fig = evoked.plot(show=False)
    fig.savefig(output_dir / "erp_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    results["evoked_data"] = evoked.get_data().tolist()
    results["evoked_times"] = evoked.times.tolist()
    results["n_epochs"] = len(epochs)  # 现在可以正常获取长度了

    return results


def save_overall_results(analysis_results, output_dir):
    """保存总体分析结果"""

    # 保存为JSON
    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        # 转换numpy数组为列表以便JSON序列化
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(
                obj,
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        json.dump(convert_for_json(analysis_results), f, ensure_ascii=False, indent=2)

    # 生成文本报告
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("多模态生理信号分析报告\n")
        f.write("=" * 50 + "\n\n")

        if "eeg" in analysis_results:
            eeg_info = analysis_results["eeg"]["basic_info"]
            f.write("EEG分析:\n")
            f.write(f"  采样率: {eeg_info['sampling_rate']} Hz\n")
            f.write(f"  持续时间: {eeg_info['duration']:.2f} 秒\n")
            f.write(f"  EEG通道数: {len(eeg_info['eeg_channels'])}\n")
            f.write(f"  AUX通道数: {len(eeg_info['aux_channels'])}\n")
            f.write(
                f"  电极位置设置: {'成功' if eeg_info['electrode_setup_success'] else '失败'}\n\n"
            )

            if "eeg_analysis" in analysis_results["eeg"]:
                eeg_analysis = analysis_results["eeg"]["eeg_analysis"]
                if "ica_success" in eeg_analysis:
                    f.write(
                        f"  ICA分析: {'成功' if eeg_analysis['ica_success'] else '失败'}\n"
                    )
                    if eeg_analysis["ica_success"]:
                        f.write(f"  ICA组件数: {eeg_analysis['ica_n_components']}\n")

        if "fnirs" in analysis_results and "error" not in analysis_results["fnirs"]:
            nirs_info = analysis_results["fnirs"]["basic_info"]
            f.write("fNIRS分析:\n")
            f.write(f"  采样率: {nirs_info['sampling_rate']} Hz\n")
            f.write(f"  原始通道数: {len(nirs_info['raw_channels'])}\n")
            f.write(f"  HBO通道数: {len(nirs_info['hbo_channels'])}\n")
            f.write(f"  HBR通道数: {len(nirs_info['hbr_channels'])}\n\n")

        # 添加其他统计信息
        if "eeg" in analysis_results and "ecg_analysis" in analysis_results["eeg"]:
            ecg_info = analysis_results["eeg"]["ecg_analysis"]
            if "mean_heart_rate" in ecg_info:
                f.write("ECG分析:\n")
                f.write(
                    f"  平均心率: {ecg_info.get('mean_heart_rate', 'N/A'):.1f} BPM\n"
                )
                f.write(f"  HRV RMSSD: {ecg_info.get('hrv_rmssd', 'N/A'):.2f} ms\n\n")

        if "eeg" in analysis_results and "eog_analysis" in analysis_results["eeg"]:
            eog_info = analysis_results["eeg"]["eog_analysis"]
            if "blink_rate" in eog_info:
                f.write("EOG分析:\n")
                f.write(
                    f"  眨眼率: {eog_info.get('blink_rate', 'N/A'):.1f} 次/分钟\n\n"
                )

        if "eeg" in analysis_results and "gsr_analysis" in analysis_results["eeg"]:
            gsr_info = analysis_results["eeg"]["gsr_analysis"]
            if "scr_count" in gsr_info:
                f.write("GSR分析:\n")
                f.write(f"  SCR事件数: {gsr_info.get('scr_count', 'N/A')}\n")


def inspect_xdf(xdf_file_path: Path, output_path: Path):
    if not xdf_file_path.exists():
        raise FileNotFoundError(f"文件不存在: {xdf_file_path}")

    print(f"读取 XDF 文件: {xdf_file_path}\n")

    streams, header = load_xdf(str(xdf_file_path), dejitter_timestamps=False)

    # ----------------------------
    # 打印文件级别信息
    # ----------------------------
    print("====== 文件级别信息 ======")
    for k, v in header.items():
        print(f"{k}: {v}")
    print()

    # 用于导出 JSON 的结构
    json_dict = {"file_header": header, "streams": []}

    # ----------------------------
    # 打印 Stream 信息并构建 JSON
    # ----------------------------
    print("====== Stream 列表 ======")
    for idx, stream in enumerate(streams):
        info = stream["info"]

        name = info.get("name", ["<unknown>"])[0]
        stype = info.get("type", ["<unknown>"])[0]
        srate = info.get("sample_rate", ["<unknown>"])[0]
        channels = info.get("channel_count", ["<unknown>"])[0]

        print(f"[{idx}] {name}")
        print(f"  类型(type):        {stype}")
        print(f"  采样率(sample_rate): {srate}")
        print(f"  通道数(channels):   {channels}")
        print("  其他元数据(info):")
        for key, value in info.items():
            if key in ["name", "type", "sample_rate", "channel_count"]:
                continue
            print(f"    {key}: {value}")

        print("-" * 50)

        # 写入 JSON 结构
        json_dict["streams"].append(
            {
                "index": idx,
                "name": name,
                "type": stype,
                "sample_rate": srate,
                "channel_count": channels,
                "full_info": info,
            }
        )

    # ----------------------------
    # 导出 JSON
    # ----------------------------
    json_path = output_path
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)

    print(f"\nJSON 已导出到: {json_path}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    xdf_file_path = Path(input("请输入XDF文件路径:\n").strip('"')).resolve()

    OmegaConf.resolve(cfg)

    result_dir = Path(cfg.result_dir) / xdf_file_path.stem
    result_dir.mkdir(parents=True, exist_ok=True)

    if "inspect" in cfg and cfg.inspect:
        inspect_xdf(xdf_file_path, result_dir / "inspect.json")
        return
    analyze_xdf_data(xdf_file_path, result_dir)


if __name__ == "__main__":
    main()
