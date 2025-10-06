import random
from pathlib import Path

from PIL import Image
from psychopy import core, event, visual
from pylsl import StreamInfo, StreamOutlet

orbitary_keys = (
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


def send_marker(lsl_outlet: StreamOutlet, marker: str, timestamp: float | None = None):
    """向 LSL 发送 marker"""
    if lsl_outlet is not None:
        lsl_outlet.push_sample([marker], timestamp)


def switch_keyboard_layout(layout: str = "en-US"):
    """切换到指定的键盘布局"""
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
    """解析刺激字符串"""
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


if __name__ == "__main__":
    pass
