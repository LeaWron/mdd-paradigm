import ctypes
import logging
import math
import random
import tkinter as tk
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import win32api
import win32con
from PIL import Image
from psychopy import event, gui, monitors, visual
from pyglet import canvas
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


# === lsl === #
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
    is_pre: bool = False,
):
    """
    向 LSL 发送 marker
    Args:
        lsl_outlet (StreamOutlet): LSL 输出流
        marker (str): 要发送的 marker 字符串
        timestamp (float | None, optional): 时间戳. Defaults to None.
        is_pre (bool, optional): 预实验时不发送 marker. Defaults to False.
    """
    if lsl_outlet is None:
        return
    if not is_pre:
        lsl_outlet.push_sample(
            [str(marker)], timestamp if timestamp is not None else local_clock()
        )


# === session === #
def switch_keyboard_layout(layout: str = "en-US"):
    # 加载 user32.dll
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    match layout:
        case "en-US":
            # HKL 对应输入法标识符：0x04090409 = en-US 键盘布局
            HKL_NEXT = 0x04090409
        case "zh-CN":
            # HKL 对应输入法标识符：0x08040804 = zh-CN 键盘布局
            HKL_NEXT = 0x08040804
        case _:
            raise ValueError(f"不支持的键盘布局 {layout}")

    # 切换输入法
    user32.ActivateKeyboardLayout(HKL_NEXT, 0)


# === experiment === #
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


def into_stim_str(stim: Path) -> str:
    """
    将刺激路径转换为项目内相对路径字符串

    Args:
        stim (str | Path): 刺激字符串或路径
    Returns:
        str: 刺激路径字符串
    """
    stim_dir = Path(__file__).parent / "stims"
    stim = str(stim.relative_to(stim_dir))
    return stim


def adapt_image_stim_size(win: visual.Window, stim_path: Path, max_height: float = 2.0):
    """
    调整图像刺激大小, 使图像在保持宽高比的前提下, 能被屏幕正好容纳下
    Args:
        win (visual.Window): 窗口对象
        stim_path (Path): 图像刺激路径
        max_height (float, optional): 最大高度, 单位为屏幕高度. Defaults to 2.0.

    Returns:
        tuple: (stim_height, aspect_ratio), 图像刺激高度, 宽高比
    """
    screen_width, screen_height = win.size
    img = Image.open(stim_path)
    width, height = img.size

    aspect_ratio = (width * screen_height) / (height * screen_width)

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


class ScreenUtils:
    def __init__(self, show: bool = False):
        if show:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        self.logger = logging.getLogger("ScreenUtils")

    def set_dpi_awareness(self, dpi_awareness: int = 2):
        """设置应用程序的DPI感知级别，以正确处理高DPI显示器"""
        # 0: PROCESS_DPI_UNAWARE
        # 1: PROCESS_SYSTEM_DPI_AWARE
        # 2: PROCESS_PER_MONITOR_DPI_AWARE
        try:
            # 尝试设置每监视器DPI感知（Windows 8.1+）
            ctypes.windll.shcore.SetProcessDpiAwareness(dpi_awareness)
        except Exception as e:
            self.logger.error(f"Error setting DPI awareness: {e}")

            try:
                # 回退到系统DPI感知（Windows Vista+）
                ctypes.windll.user32.SetProcessDpiAwarenessContext(
                    -4
                )  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            except Exception as e:
                self.logger.error(f"Error setting DPI awareness context: {e}")
                try:
                    # 更老的回退方案
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception as e:
                    self.logger.error(f"Error setting DPI aware: {e}")

    def get_dpi_for_monitor(self, hmonitor):
        """获取指定监视器的DPI"""
        try:
            dpi_x = ctypes.c_uint()
            dpi_y = ctypes.c_uint()
            ctypes.windll.shcore.GetDpiForMonitor(
                hmonitor, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y)
            )
            return dpi_x.value, dpi_y.value
        except Exception as e:
            self.logger.error(f"Error getting DPI for monitor: {e}")
            return 96, 96  # 默认DPI

    def get_display_devices_dpi_aware(self):
        """改进版的显示器信息获取函数，考虑DPI缩放"""
        # 设置DPI感知
        self.set_dpi_awareness()

        import ctypes
        from ctypes import wintypes

        # 定义回调函数类型
        MONITORENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.wintypes.HMONITOR,
            ctypes.wintypes.HDC,
            ctypes.POINTER(wintypes.RECT),
            ctypes.wintypes.LPARAM,
        )

        # 存储显示器信息
        screens = []

        def enum_callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
            # 使用 win32api 获取详细信息
            try:
                monitor_info = win32api.GetMonitorInfo(hMonitor)

                # 获取DPI信息
                dpi_x, dpi_y = self.get_dpi_for_monitor(hMonitor)

                # 添加DPI信息到monitor_info
                monitor_info["DPI"] = (dpi_x, dpi_y)
                monitor_info["ScalingFactor"] = (
                    dpi_x / 96.0,
                    dpi_y / 96.0,
                )  # 相对于标准DPI的缩放因子

                screens.append(monitor_info)
            except Exception as e:
                self.logger.error(f"Error getting monitor info: {e}")
                pass
            return True

        # 创建回调函数
        callback = MONITORENUMPROC(enum_callback)

        # 枚举所有显示器
        ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, 0)

        # 打印所有显示器信息
        for i, screen in enumerate(screens):
            self.logger.debug(f"显示器 {i + 1}:")
            self.logger.debug(f"  设备名称: {screen['Device']}")
            self.logger.debug(f"  是否主要显示器: {screen['Flags'] == 1}")
            self.logger.debug(
                f"  工作区域: {screen['Work']}"
            )  # 工作区域（不包含任务栏）
            self.logger.debug(f"  显示器矩形: {screen['Monitor']}")  # 整个显示器区域
            self.logger.debug(f"  DPI: {screen['DPI']}")
            self.logger.debug(f"  缩放因子: {screen['ScalingFactor']}")
            self.logger.debug("")

        return screens

    def get_screen_pyglet_dpi_aware(self):
        """改进版的pyglet屏幕信息获取函数，考虑DPI缩放"""
        # 设置DPI感知
        self.set_dpi_awareness()

        try:
            screens = canvas.get_display().get_screens()
            for i, screen in enumerate(screens):
                self.logger.debug(f"屏幕 {i + 1}:")
                self.logger.debug(f"  x: {screen.x}, y: {screen.y}")
                self.logger.debug(f"  width: {screen.width}, height: {screen.height}")
                # 注意：在高DPI显示器上，这些值可能是逻辑像素而不是物理像素
                # 可能需要结合DPI信息来计算实际物理尺寸
            return screens
        except Exception as e:
            self.logger.error(f"Error getting pyglet screens: {e}")
            return None

    def get_physical_screen_size(self, hmonitor):
        """获取监视器的物理尺寸（毫米）"""
        try:
            # 使用GetMonitorInfo获取监视器信息
            monitor_info = win32api.GetMonitorInfo(hmonitor)

            # 获取设备名称
            device_name = monitor_info["Device"]

            # 获取显示设置
            dev_mode = win32api.EnumDisplaySettings(
                device_name, win32con.ENUM_CURRENT_SETTINGS
            )

            # 获取DPI
            dpi_x, dpi_y = self.get_dpi_for_monitor(hmonitor)

            # 计算物理尺寸（英寸）
            physical_width_inch = dev_mode.PelsWidth / dpi_x
            physical_height_inch = dev_mode.PelsHeight / dpi_y

            # 转换为毫米（1英寸 = 25.4毫米）
            physical_width_mm = physical_width_inch * 25.4
            physical_height_mm = physical_height_inch * 25.4

            return physical_width_mm, physical_height_mm
        except Exception as e:
            self.logger.error(f"Error getting physical screen size: {e}")
            return None, None

    def get_comprehensive_display_info(self):
        """综合显示信息测试，包含DPI和物理尺寸"""
        # 设置DPI感知
        self.set_dpi_awareness()

        import ctypes

        # 定义回调函数类型
        MONITORENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.wintypes.HMONITOR,
            ctypes.wintypes.HDC,
            ctypes.POINTER(wintypes.RECT),
            ctypes.wintypes.LPARAM,
        )

        # 存储显示器信息
        monitors = []

        def enum_callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
            try:
                # 基本监视器信息
                monitor_info = win32api.GetMonitorInfo(hMonitor)

                # DPI信息
                dpi_x, dpi_y = self.get_dpi_for_monitor(hMonitor)

                # 物理尺寸
                phys_width, phys_height = self.get_physical_screen_size(hMonitor)

                # 组合信息
                comprehensive_info = {
                    "basic": monitor_info,
                    "dpi": (dpi_x, dpi_y),
                    "scaling_factor": (dpi_x / 96.0, dpi_y / 96.0),
                    "physical_size_mm": (phys_width, phys_height),
                }

                monitors.append(comprehensive_info)
            except Exception as e:
                self.logger.error(f"Error getting comprehensive monitor info: {e}")
            return True

        # 创建回调函数
        callback = MONITORENUMPROC(enum_callback)

        # 枚举所有显示器
        ctypes.windll.user32.EnumDisplayMonitors(None, None, callback, 0)

        # 打印所有显示器信息
        for i, monitor in enumerate(monitors):
            basic_info = monitor["basic"]
            dpi = monitor["dpi"]
            scaling = monitor["scaling_factor"]
            physical_size = monitor["physical_size_mm"]

            self.logger.debug(f"=== 显示器 {i + 1} ===")
            self.logger.debug(f"设备名称: {basic_info['Device']}")
            self.logger.debug(f"工作区域: {basic_info['Work']}")
            self.logger.debug(f"是否主要显示器: {basic_info['Flags'] == 1}")
            self.logger.debug(f"显示器矩形: {basic_info['Monitor']}")
            self.logger.debug(f"DPI: {dpi}")
            self.logger.debug(f"缩放因子: {scaling[0]:.2f}x")
            if physical_size[0] and physical_size[1]:
                self.logger.debug(
                    f"物理尺寸: {physical_size[0]:.1f}mm x {physical_size[1]:.1f}mm"
                )
        self.set_dpi_awareness(0)

        return monitors


class PsychopyDisplaySelector:
    def __init__(self):
        self.screen_utils = ScreenUtils()
        self.selected_screen = None
        self.screens_info = []

    def get_all_screens_info(self):
        """获取所有屏幕的详细信息"""
        # 获取综合显示信息
        comprehensive_info = self.screen_utils.get_comprehensive_display_info()
        self.screens_info = comprehensive_info
        return comprehensive_info

    def create_display_dialog(self):
        """创建Psychopy GUI对话框用于选择显示器"""
        # 获取所有屏幕信息
        screens_info = self.get_all_screens_info()

        # 创建对话框
        dlg = gui.Dlg(title="显示器选择")
        dlg.addText("pos代表该显示器左上角与主显示器的相对位置")
        dlg.addText("请选择要使用的显示器:")

        # 为每个屏幕添加信息
        screen_choices = []
        for i, screen_info in enumerate(screens_info):
            basic_info = screen_info["basic"]

            screen_rect = basic_info["Monitor"]
            width = screen_rect[2] - screen_rect[0]
            height = screen_rect[3] - screen_rect[1]

            # 创建屏幕描述
            screen_desc = (
                f"屏幕 {i + 1}: "
                f"{width}x{height} | "
                f"pos: {(screen_rect[0], screen_rect[1])}"
                f"主要: {'是' if basic_info['Flags'] == 1 else '否'}"
            )

            screen_choices.append(screen_desc)

        # 添加选择字段
        dlg.addField("显示器:", choices=screen_choices)

        # 显示对话框
        ok_data = dlg.show()

        if dlg.OK:
            selected_index = (
                screen_choices.index(ok_data[0]) if ok_data[0] in screen_choices else 0
            )
            self.selected_screen = selected_index

        else:
            return None
        return self.selected_screen

    def preview_selected_screen(self, screen_index):
        """预览选中的屏幕"""
        if screen_index < 0 or screen_index >= len(self.screens_info):
            print("无效的屏幕索引")
            return

        try:
            # 获取屏幕信息
            screen_info = self.screens_info[screen_index]
            basic_info = screen_info["basic"]
            dpi = screen_info["dpi"]
            scaling = screen_info["scaling_factor"]

            # 获取屏幕矩形
            screen_rect = basic_info["Monitor"]
            width = screen_rect[2] - screen_rect[0]
            height = screen_rect[3] - screen_rect[1]

            # 创建预览窗口 - 使用指定的屏幕
            win = visual.Window(
                size=(min(800, width), min(600, height)),  # 限制窗口大小
                pos=(0, 0),
                screen=screen_index,
                fullscr=False,
                color="grey",
                units="norm",
            )

            # 显示屏幕信息
            info_text = (
                f"屏幕 {screen_index + 1} 预览\n\n"
                f"分辨率: {width} x {height}\n"
                f"DPI: {dpi[0]} x {dpi[1]}\n"
                f"缩放因子: {scaling[0]:.2f}x\n"
                f"主要显示器: {'是' if basic_info['Flags'] == 1 else '否'}\n\n"
                f"按任意键关闭预览"
            )

            # 创建文本刺激
            text_stim = visual.TextStim(
                win=win,
                text=info_text,
                pos=(0, 0),
                color="white",
                height=0.05,
                wrapWidth=1.8,
            )

            # 绘制并显示
            text_stim.draw()
            win.flip()

            # 等待按键关闭
            event.waitKeys()
            win.close()

        except Exception as e:
            print(f"预览屏幕时出错: {e}")

            # 创建错误提示窗口
            error_win = visual.Window(
                size=(400, 200),
                color="black",
                screen=0,  # 在主屏幕显示错误
                fullscr=False,
            )

            error_text = visual.TextStim(
                win=error_win,
                text=f"无法创建预览窗口:\n{str(e)}",
                color="white",
                height=0.06,
            )

            error_text.draw()
            error_win.flip()
            event.waitKeys()
            error_win.close()

    def select_and_preview(self):
        """选择显示器并预览"""
        # 显示选择对话框
        while True:
            selected_index = self.create_display_dialog()
            if selected_index is None:
                break
            sub_dlg = gui.Dlg(title="确认选择")
            sub_dlg.addField(
                label="是否预览选中屏幕位置?否则直接确认",
                initial=False,
                key="preview",
            )
            sub_ok = sub_dlg.show()
            if sub_ok["preview"]:
                self.preview_selected_screen(selected_index)
                sub_dlg = gui.Dlg(title="确认选择")
                sub_dlg.addField(
                    label=f"是否选择屏幕 {selected_index + 1}?",
                    initial=True,
                    key="confirm",
                )
                sub_ok = sub_dlg.show()
                if sub_ok["confirm"]:
                    break
            else:
                break
        if selected_index is None:
            print("用户取消了选择")
        return selected_index

    def get_selected_screen_window_params(self):
        """获取选中屏幕的窗口参数"""
        if self.selected_screen is None:
            # 如果没有选择，返回默认参数
            return None

        # 获取选中屏幕的信息
        screen_info = self.screens_info[self.selected_screen]
        needed_params = {}
        # 分辨率
        needed_params["pix_size"] = (
            screen_info["basic"]["Monitor"][2] - screen_info["basic"]["Monitor"][0],
            screen_info["basic"]["Monitor"][3] - screen_info["basic"]["Monitor"][1],
        )
        # 物理尺寸
        needed_params["phys_size_mm"] = screen_info["physical_size_mm"]

        # 返回窗口参数
        return needed_params


def create_monitor(
    screen_info: dict, my_monitor: str = "subMonitor", logger: logging.Logger = None
) -> monitors.Monitor:
    """
    创建新的显示器配置

    Returns:
        psychopy.monitors.Monitor: 创建的显示器对象
    """
    dlg = gui.Dlg(title="创建显示器配置")
    dlg.addField(label="显示器名称", initial=my_monitor, key="name")
    dlg.addField(label="分辨率", initial=screen_info["pix_size"], key="pix_size")
    dlg.addField(
        label="物理尺寸(mm)", initial=screen_info["phys_size_mm"], key="phys_size_mm"
    )
    dlg.addField(label="与人的距离(cm)", initial=60, key="distance")
    ok = dlg.show()
    if ok is None:
        return None

    monitor = monitors.Monitor(name=ok["name"])
    monitor.setSizePix(ok["pix_size"])
    monitor.setWidth(ok["phys_size_mm"][0] / 10)
    monitor.setDistance(ok["distance"])
    monitor.save()
    if logger:
        logger.info(f"创建新的显示器配置: {ok['name']}")
    return monitor


def select_monitor(
    screen_info: dict, my_monitor: str = "subMonitor", logger: logging.Logger = None
) -> monitors.Monitor:
    """
    选择要使用的显示器

    Returns:
        psychopy.monitors.Monitor: 选择的显示器对象
    """
    available_monitors = monitors.getAllMonitors()
    if my_monitor not in available_monitors:
        dlg = gui.Dlg(
            title="错误",
        )
        dlg.addText(
            f"指定的显示器配置 {my_monitor} 不存在，可用显示器列表中只有 {available_monitors}",
        )
        dlg.addField(label="是否创建新的显示器配置?", initial=True, key="create")
        ok = dlg.show()
        if ok and ok["create"]:
            return create_monitor(screen_info, my_monitor, logger)
        else:
            gui.infoDlg(title="通知", prompt="将会在已有显示器配置中选择")

    dlg = gui.Dlg(title="选择显示器配置")
    dlg.addField(
        label="请选择要使用的显示器配置",
        choices=available_monitors,
        initial=available_monitors[0],
        key="monitor",
    )
    ok = dlg.show()
    if ok is None:
        dlg = gui.Dlg(title="错误")
        dlg.addText("用户取消了选择显示器配置")
        dlg.addField(label="是否创建并使用新的显示器配置?", initial=True, key="create")
        ok = dlg.show()
        if ok and ok["create"]:
            return create_monitor(screen_info, "", logger)
    monitor = ok["monitor"]

    return monitor


# === misc === #
# 每个范式只生成一次, 所以简单点
def generate_trial_sequence(
    n_blocks: int,
    n_trials_per_block: int,
    max_seq_same: int = 2,
    all_occur: bool = True,
    stim_list: list = None,
    stim_weights: list = None,
    seed: int = None,
):
    from collections import Counter, defaultdict
    from pprint import pprint

    def check_seq(seq: list, max_seq_same: int, all_occur: bool) -> bool:
        current_count = 0
        prev_choice = None
        appeared = set()
        for item in seq:
            if item == prev_choice:
                current_count += 1
            else:
                current_count = 1
                prev_choice = item
            if current_count > max_seq_same:
                return False
            appeared.add(item)
        if all_occur and len(appeared) != len(stim_list):
            return False
        return True

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)

    stim_sequences = defaultdict(list)
    for block_index in range(n_blocks):
        while True:
            seq: list = rng.choice(
                stim_list, size=n_trials_per_block, replace=True, p=stim_weights
            ).tolist()
            if check_seq(seq, max_seq_same, all_occur):
                stim_sequences[block_index].extend(seq)
                break

    stim_sequences = dict(stim_sequences)

    return stim_sequences


def update_trial(one_trial_data: dict[str, Any], one_block_data: dict[str, list]):
    for key, value in one_trial_data.items():
        one_block_data[key].append(value)
        one_trial_data[key] = None


def update_block(one_block_data: dict[str, list], data_to_save: dict[str, list]):
    for key, value in one_block_data.items():
        data_to_save[key].extend(value)
        one_block_data[key] = []


def save_csv_data(
    data: dict[str, list], file_name: str | Path, participant_type: str = None
):
    """
    将实验数据保存为 CSV 文件

    Args:
        data (dict[str, list]): key 为列名, value 为该列的数据（长度需一致）
        file_name (str | Path): 保存文件名, 会保存到根目录的 data 文件夹下的当前日期文件夹, 会自动添加 .csv 后缀
    """
    base_path = Path(__file__).parent.parent / "data"
    date_folder = base_path / datetime.now().strftime("%Y-%m-%d")
    # 日期文件夹
    date_folder.mkdir(parents=True, exist_ok=True)

    file_name_2 = None
    if participant_type is not None:
        file_name_2 = (
            base_path / participant_type.lower() / f"{file_name}"
        ).with_suffix(".csv")
        file_name_2.parent.mkdir(parents=True, exist_ok=True)

    file_name = (date_folder / f"{file_name}").with_suffix(".csv")

    max_len = max(len(v) for v in data.values()) if data else 0
    for _, v in data.items():
        if len(v) < max_len:
            v.extend([None] * (max_len - len(v)))

    df = pl.DataFrame(data, strict=False)

    if file_name.exists():
        with open(file_name, "a", encoding="utf-8", newline="") as f:
            df.write_csv(f, include_header=False)
    else:
        df.write_csv(file_name)

    if file_name_2 is not None:
        if file_name_2.exists():
            with open(file_name_2, "a", encoding="utf-8", newline="") as f:
                df.write_csv(f, include_header=False)
        else:
            df.write_csv(file_name_2)


def setup_default_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    )
    return logging.getLogger("default")


def get_audio_devices():
    import sounddevice as sd

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    clean = []
    for d in devices:
        if d["max_output_channels"] <= 0:
            continue
        api_name = hostapis[d["hostapi"]]["name"]
        if any(skip in api_name for skip in ["MME", "DirectSound", "ASIO", "WDM-KS"]):
            continue
        clean.append(f"{d['name']}")
    return sorted(list(set(clean)))


if __name__ == "__main__":
    pass
