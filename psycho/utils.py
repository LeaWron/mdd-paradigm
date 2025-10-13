import ctypes
import math
import random
import tkinter as tk
from pathlib import Path

import polars as pl
from PIL import Image
from pylsl import StreamInfo, StreamOutlet, local_clock

arbitary_keys = (
    [chr(i) for i in range(32, 127)]
    + ["return", "space"]
    + ["num0", "num1", "num2", "num3", "num4", "num5", "num6", "num7", "num8", "num9"]
    + [
        "comma",
        "period",
        "slash",
        "semicolon",
        "apostrophe",
        "bracketleft",
        "bracketright",
        "backslash",
    ]
    + ["left", "right", "up", "down"]
)


def init_lsl(
    name,
    stream_info_type: str = "Markers",
    channel_count: int = 1,
    channel_format: str = "string",
    source_id: str | None = None,
):
    # 定义一个 marker stream
    info = StreamInfo(
        name=name,
        type=stream_info_type,
        channel_count=channel_count,
        channel_format=channel_format,
        source_id=source_id if source_id is not None else f"{name}_{7758}",
    )
    lsl_outlet = StreamOutlet(info)
    return lsl_outlet


def send_marker(
    lsl_outlet: StreamOutlet,
    marker: str,
    timestamp: float | None = None,
):
    """
    向 LSL 发送 marker
    Args:
        lsl_outlet (StreamOutlet): LSL 输出流
        marker (str): 要发送的 marker 字符串
        timestamp (float | None, optional): 时间戳. Defaults to None.
    """
    if lsl_outlet is None:
        return
    lsl_outlet.push_sample([marker], timestamp if timestamp is not None else local_clock())


def switch_keyboard_layout(layout: str = "en-US"):
    import ctypes

    # 加载 user32.dll
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    # HKL 对应输入法标识符：0x04090409 = en-US 键盘布局
    HKL_NEXT = 0x04090409

    # 切换输入法
    user32.ActivateKeyboardLayout(HKL_NEXT, 0)


def get_isi(lower_bound: float = 0.5, upper_bound: float = 1.0) -> float:
    """
    获取随机的 ISI 间隔, 单位为秒

    Args:
        lower_bound (float, optional): 下限, 单位为秒. Defaults to 0.5.
        upper_bound (float, optional): 上限, 单位为秒. Defaults to 1.0.

    Returns:
        float: 随机的 ISI 间隔, 单位为秒
    """
    return random.uniform(lower_bound * 1000, upper_bound * 1000) / 1000


def parse_stim_path(stim: str) -> Path:
    stim = stim.strip()
    stim_dir = Path(__file__).parent / "stims"
    stim_path = stim_dir / stim
    if not stim_path.exists():
        raise FileNotFoundError(f"刺激文件 {stim_path} 不存在")
    return stim_path


def adapt_image_stim_size(stim_path: Path, max_height: float = 2.0):
    """
    调整图像刺激大小, 使图像在保持宽高比的前提下, 能被屏幕正好容纳下
    Args:
        stim_path (Path): 图像刺激路径
        max_height (float, optional): 最大高度, 单位为屏幕高度. Defaults to 2.0.

    Returns:
        tuple: (stim_height, aspect_ratio), 图像刺激高度, 宽高比
    """
    img = Image.open(stim_path)
    width, height = img.size

    aspect_ratio = width / height

    if width < height:
        stim_height = max_height
    else:
        stim_height = max_height / aspect_ratio
    return stim_height, aspect_ratio


def cm_to_unit_ratio(cm):
    """
    输入: cm（厘米）
    输出: 对应在屏幕坐标系下的比例值 (0.0 ~ 1.0)
         坐标系说明:
         - 屏幕中心为 0.0
         - 四个角点为 ±1
         - 不考虑象限，只返回模长比例
    """
    # --- 设置 DPI 感知模式，防止 Windows 缩放干扰 ---
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Win 8.1+
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    # --- 获取屏幕 DPI 和分辨率 ---
    root = tk.Tk()
    dpi = root.winfo_fpixels("1i")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # --- 转换 cm → px ---
    pixels = (cm / 2.54) * dpi

    # --- 计算半对角线长度（中心 → 角点 的像素距离）---
    max_radius_px = math.hypot(screen_width / 2, screen_height / 2)

    # --- 计算比例 ---
    ratio = pixels / max_radius_px

    # 限定在 [0, 1]
    if ratio < 0:
        ratio = 0.0
    elif ratio > 1:
        ratio = 1.0

    return ratio


# 每个范式只生成一次, 所以简单点
def generate_trial_sequence(
    n_blocks: int,
    n_trials_per_block: int,
    max_seq_same: int = 2,
    stim_list: list = None,
):
    from collections import defaultdict

    stim_sequences = defaultdict(list)
    for block_index in range(n_blocks):
        current_count = 0
        prev_choice = None

        while True:
            temp_seq = []
            for _ in range(n_trials_per_block):
                current_choice = random.choice(stim_list)
                if current_choice == prev_choice:
                    current_count += 1
                else:
                    current_count = 1
                    prev_choice = current_choice
                if current_count > max_seq_same:
                    temp_seq.clear()
                    break
                temp_seq.append(current_choice)
            if len(temp_seq) == n_trials_per_block:
                break

        stim_sequences[block_index].extend(temp_seq)

    return stim_sequences


def save_csv_data(data: dict[str, list], file_path: str | Path):
    """
    将实验数据保存为 CSV 文件

    Args:
        data (dict[str, list]): key 为列名, value 为该列的数据（长度需一致）
        file_path (str | Path): 保存路径
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    max_len = max(len(v) for v in data.values()) if data else 0
    for _, v in data.items():
        if len(v) < max_len:
            v.extend([None] * (max_len - len(v)))

    df = pl.DataFrame(data)

    if file_path.exists():
        with open(file_path, "a", encoding="utf-8", newline="") as f:
            df.write_csv(f, include_header=False)
    else:
        df.write_csv(file_path)


if __name__ == "__main__":
    pass
