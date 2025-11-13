import json
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mne
import numpy as np
import pyxdf
import seaborn as sns
from mne.io import RawArray
from omegaconf import DictConfig, OmegaConf
from pyxdf import load_xdf


# TODO: 使用其他库来解析 xdf 文件
def parse_xdf(xdf_file_path: str, output_dir: Path = "../results"):
    """
    解析XDF文件中的多模态生物信号数据并生成波形图

    参数:
    xdf_file_path: XDF文件路径
    output_dir: 输出目录（可选，默认为hydra的配置)
    """

    # 创建子目录保存波形图
    plots_dir = output_dir / "waveform_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载XDF文件
    print(f"正在加载XDF文件: {xdf_file_path}")
    try:
        streams, fileheader = load_xdf(xdf_file_path)
        print(f"成功加载XDF文件，包含 {len(streams)} 个数据流")
    except Exception as e:
        print(f"加载XDF文件失败: {e}")
        return None

    # 2. 分类不同的数据流
    modality_streams = {
        "EEG": [],
        "ECG": [],
        "EOG": [],
        "GSR": [],
        "fNIRS": [],
        "other": [],
    }

    for i, stream in enumerate(streams):
        stream_info = stream["info"]
        stream_name = (
            stream_info["name"][0].lower() if stream_info["name"] else f"stream_{i}"
        )

        # 根据流名称识别数据类型
        if "eeg" in stream_name:
            modality_streams["EEG"].append((i, stream))
        elif "ecg" in stream_name or "ekg" in stream_name:
            modality_streams["ECG"].append((i, stream))
        elif "eog" in stream_name:
            modality_streams["EOG"].append((i, stream))
        elif "gsr" in stream_name or "eda" in stream_name or "skin" in stream_name:
            modality_streams["GSR"].append((i, stream))
        elif "fnirs" in stream_name or "nirs" in stream_name:
            modality_streams["fNIRS"].append((i, stream))
        else:
            modality_streams["other"].append((i, stream))

    # 3. 处理每种数据类型并生成波形图
    results = {}

    for modality, streams_list in modality_streams.items():
        if not streams_list:
            print(f"未找到{modality}数据流")
            continue

        print(f"\n处理{modality}数据，找到{len(streams_list)}个流")

        for stream_idx, (original_idx, stream) in enumerate(streams_list):
            try:
                # 提取流数据
                data = np.array(stream["time_series"]).T
                timestamps = np.array(stream["time_stamps"])

                # 获取流信息
                stream_info = stream["info"]
                nominal_srate = (
                    float(stream_info["nominal_srate"][0])
                    if stream_info["nominal_srate"]
                    else 100
                )

                # 创建通道名称
                if "desc" in stream_info and stream_info["desc"]:
                    try:
                        ch_names = [
                            ch["label"][0]
                            for ch in stream_info["desc"][0]["channels"][0]["channel"]
                        ]
                    except Exception as _:
                        ch_names = [f"ch_{i}" for i in range(data.shape[0])]
                else:
                    ch_names = [f"ch_{i}" for i in range(data.shape[0])]

                # 设置通道类型
                if modality == "EEG":
                    ch_types = ["eeg"] * len(ch_names)
                elif modality == "ECG":
                    ch_types = ["ecg"] * len(ch_names)
                elif modality == "EOG":
                    ch_types = ["eog"] * len(ch_names)
                elif modality == "GSR":
                    ch_types = ["gsr"] * len(ch_names)
                elif modality == "fNIRS":
                    ch_types = ["fnirs"] * len(ch_names)
                else:
                    ch_types = ["misc"] * len(ch_names)

                # 创建MNE Raw对象[1](@ref)
                info = mne.create_info(
                    ch_names=ch_names, sfreq=nominal_srate, ch_types=ch_types
                )

                raw = RawArray(data, info)

                # 保存处理后的数据
                stream_key = f"{modality}_stream_{stream_idx}"
                results[stream_key] = {
                    "raw": raw,
                    "timestamps": timestamps,
                    "original_stream_idx": original_idx,
                }

                # 生成波形图
                generate_waveform_plot(
                    raw,
                    timestamps,
                    modality,
                    stream_idx,
                    ch_names,
                    plots_dir,
                    nominal_srate,
                )

            except Exception as e:
                print(f"处理{modality}流{stream_idx}时出错: {e}")

    print(f"\n所有波形图已保存至: {plots_dir}")
    return results


def generate_waveform_plot(
    raw: RawArray,
    timestamps: np.ndarray,
    modality: str,
    stream_idx: int,
    ch_names: list[str],
    output_dir: Path,
    sfreq: float,
):
    """
    生成生物信号波形图
    """
    # 设置图形大小和样式
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    # 计算时间轴
    duration = len(timestamps) / sfreq if sfreq > 0 else len(timestamps)
    time_axis = np.linspace(0, duration, len(timestamps))

    # 获取数据
    data, times = raw[:, :]

    # 根据通道数量决定显示方式
    n_channels = data.shape[0]

    if n_channels <= 8:
        # 少通道：分别绘制每个通道
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2 * n_channels))
        if n_channels == 1:
            axes = [axes]

        for i, (ax, channel_data) in enumerate(zip(axes, data)):
            ax.plot(time_axis[: len(channel_data)], channel_data, linewidth=1)
            ax.set_ylabel(
                f"{ch_names[i]}\n(μV)"
                if modality in ["EEG", "ECG", "EOG"]
                else "Amplitude"
            )
            ax.grid(True, alpha=0.3)

            if i == n_channels - 1:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])

        plt.suptitle(f"{modality} Stream {stream_idx} - Multi-channel View", y=1.02)

    else:
        # 多通道：使用MNE的绘图功能[1](@ref)
        try:
            _ = raw.plot(
                n_channels=min(10, n_channels),
                duration=min(10, duration),
                scalings="auto",
                show=False,
                title=f"{modality} Stream {stream_idx}",
            )
        except Exception as e:
            print(f"绘制{modality}流{stream_idx}时出错: {e}")
            # 备用方案：绘制前几个通道
            n_show = min(6, n_channels)
            _, axes = plt.subplots(n_show, 1, figsize=(12, 2 * n_show))
            if n_show == 1:
                axes = [axes]

            for i, ax in enumerate(axes):
                ax.plot(time_axis[: len(data[i])], data[i], linewidth=1)
                ax.set_ylabel(f"{ch_names[i]}\nAmplitude")
                ax.grid(True, alpha=0.3)

                if i == n_show - 1:
                    ax.set_xlabel("Time (s)")
                else:
                    ax.set_xticklabels([])

            plt.suptitle(
                f"{modality} Stream {stream_idx} - First {n_show} Channels", y=1.02
            )

    plt.tight_layout()

    # 保存图片
    filename = f"{modality}_stream_{stream_idx}_waveform.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"已保存: {filename}")

    # 额外生成功率谱密度图（适用于EEG、ECG等）
    if modality in ["EEG", "ECG", "EOG"]:
        try:
            _ = raw.plot_psd(fmax=min(50, sfreq / 2), show=False)
            plt.title(f"{modality} Stream {stream_idx} - Power Spectral Density")
            psd_filename = f"{modality}_stream_{stream_idx}_psd.png"
            psd_filepath = os.path.join(output_dir, psd_filename)
            plt.savefig(psd_filepath, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"已保存功率谱图: {psd_filename}")
        except Exception as e:
            print(f"生成{modality}功率谱图时出错: {e}")


def analyze_multimodal_data(results):
    """
    对解析后的多模态数据进行基本分析
    """
    analysis_results = {}

    for stream_key, stream_data in results.items():
        raw = stream_data["raw"]
        modality = stream_key.split("_")[0]

        # 基本统计信息
        data, times = raw[:, :]
        channel_stats = []

        for i, channel_data in enumerate(data):
            stats = {
                "mean": np.mean(channel_data),
                "std": np.std(channel_data),
                "min": np.min(channel_data),
                "max": np.max(channel_data),
                "range": np.ptp(channel_data),
            }
            channel_stats.append(stats)

        analysis_results[stream_key] = {
            "modality": modality,
            "n_channels": data.shape[0],
            "duration": times[-1] if len(times) > 0 else 0,
            "sampling_rate": raw.info["sfreq"],
            "channel_stats": channel_stats,
        }

        print(f"\n{stream_key}分析结果:")
        print(f"  模态: {modality}")
        print(f"  通道数: {data.shape[0]}")
        print(f"  持续时间: {times[-1] if len(times) > 0 else 0:.2f}秒")
        print(f"  采样率: {raw.info['sfreq']} Hz")

    return analysis_results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    xdf_file_path = Path(input("请输入XDF文件路径:\n")).resolve()

    OmegaConf.resolve(cfg)
    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    results = parse_xdf(xdf_file_path, result_dir)

    if results:
        # 进行数据分析
        analysis = analyze_multimodal_data(results)
        with open(result_dir / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"\n数据处理完成！共处理了{len(results)}个数据流")
    else:
        print("数据处理失败，请检查文件路径和格式")


if __name__ == "__main__":
    main()
