from psychopy import core, event
from pylsl import StreamInfo, StreamOutlet


def check_exit():
    # 实时监听键盘
    keys = event.getKeys(modifiers=True)
    for key, mods in keys:
        # 如果检测到 Esc 且 Ctrl 被按着
        if key == "escape" and mods.get("shift", False):
            print("检测到 Shift+Esc，实验退出")
            core.quit()
            return True
    return False


def skip_experiment():
    """跳过当前实验"""
    print("跳过当前实验")
    keys = event.getKeys(modifiers=True)
    if "s" in keys and "shift" in keys:
        return True
    core.wait(1)
    return True


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


def send_marker(lsl_outlet: StreamOutlet, marker: str):
    """向 LSL 发送 marker"""
    if lsl_outlet is not None:
        lsl_outlet.push_sample([marker])


if __name__ == "__main__":
    pass
