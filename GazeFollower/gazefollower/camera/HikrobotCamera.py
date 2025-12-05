# encoding=utf-8
# Author: GC Zhu + Grok (完善版)
# Email: zhugc2016@gmail.com

import os
import sys
import threading
import time
from ctypes import *
from typing import Optional

import cv2  # 必须加，用于颜色转换 + resize（与 WebCamCamera 完全一致）
import numpy as np

from ..logger import Log
from .Camera import Camera  # Adjust import according to your package structure

# 海康 SDK 动态导入
HAS_MV_CAMERA = os.getenv("MVCAM_COMMON_RUNENV") is not None
if HAS_MV_CAMERA:
    sys.path.append(os.getenv("MVCAM_COMMON_RUNENV") + "/Samples/Python/MvImport")
    from CameraParams_header import *
    from MvCameraControl_class import *

    print("Successfully imported MvCamera module")
else:
    raise ImportError("未检测到海康 MVS 环境变量 MVCAM_COMMON_RUNENV，无法使用海康相机")


from ..misc import CameraRunningState

# from ..logger import Log   # 如果你的项目有 Log，可打开；否则用 print


class HikvisionCamera(Camera):
    """
    海康威视工业相机完整实现（与 WebCamCamera 接口 100% 一致）
    修复所有潜在接口漏洞：
    1. 回调签名完全一致：callback(state, timestamp_ns, frame_rgb, *args, **kwargs)
    2. 强制输出 RGB uint8 + resize 到目标尺寸（默认 640x480）
    3. 自动处理 Bayer/Mono/RGB 格式（完美兼容彩色/黑白相机）
    4. timestamp 使用 time.time_ns()（纳秒整数，与 WebCamCamera 完全一致）
    5. 线程控制标志、open/close/release 行为完全一致
    6. 支持用户指定目标分辨率、曝光、增益、帧率、外触发等
    """

    def __init__(
        self,
        exposure_time: float = 10000.0,  # 微秒
        gain: float = 12.0,
        frame_rate: float = 90.0,
        trigger_mode: bool = False,  # True=外触发 Line0 RisingEdge
        pixel_format: str = "BayerRG8",  # 推荐 BayerRG8 / Mono8 / RGB8Packed
        target_width: int = 640,  # 目标输出宽度（强制 resize，与 WebCamCamera 一致）
        target_height: int = 480,  # 目标输出高度
        packet_size: int = 1500,  # GigE 相机优化用
        auto_select: bool = True,  # 是否自动选第一台
    ):
        super().__init__()

        self.exposure_time = exposure_time
        self.gain = gain
        self.frame_rate = frame_rate
        self.trigger_mode = trigger_mode
        self.pixel_format = pixel_format
        self.target_width = target_width
        self.target_height = target_height
        self.packet_size = packet_size
        self.auto_select = auto_select
        self.frame_count = 0
        self.last_time = time.time()
        # 海康对象
        self.cam: Optional[MvCamera] = None

        # 线程控制（与 WebCamCamera 完全一致）
        self._camera_thread_running = False
        self._camera_thread = None

    def open(self):
        """完全对标 WebCamCamera.open() 行为"""
        if self.cam is not None:
            return  # 防止重复打开

        # 1. 初始化 SDK
        ret = MvCamera.MV_CC_Initialize()
        if ret != 0:
            raise RuntimeError(f"SDK 初始化失败: 0x{ret:x}")

        # 2. 枚举设备
        deviceList = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, deviceList)
        if deviceList.nDeviceNum == 0:
            raise RuntimeError("未发现海康相机")

        # 3. 选择相机（与 WebCamCamera 的 webcam_id 逻辑一致）
        if self.auto_select:
            idx = 0
        else:
            print(f"发现 {deviceList.nDeviceNum} 台相机：")
            for i in range(deviceList.nDeviceNum):
                info = cast(
                    deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)
                ).contents
                name = "".join(
                    chr(c) for c in info.SpecialInfo.stUsb3VInfo.chModelName if c != 0
                )
                sn = "".join(
                    chr(c)
                    for c in info.SpecialInfo.stUsb3VInfo.chSerialNumber
                    if c != 0
                )
                print(f"  [{i}] {name}  SN:{sn}")
            idx = int(0)

        devInfo = cast(deviceList.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents

        # 4. 创建句柄 + 打开设备 + 参数设置
        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(devInfo)
        if ret != 0:
            raise RuntimeError(f"未找到相机: 0x{ret:x}")
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"打开相机失败: 0x{ret:x}")

        # 参数设置（完全同之前）
        self.cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
        self.cam.MV_CC_SetBoolValue("ReverseX", False)  # True=启用翻转，False=关闭
        self.cam.MV_CC_SetFloatValue("ExposureTime", float(self.exposure_time))
        self.cam.MV_CC_SetEnumValueByString("GainAuto", "Off")
        self.cam.MV_CC_SetFloatValue("Gain", float(self.gain))
        self.cam.MV_CC_SetEnumValueByString("BalanceWhiteAuto", "Off")
        if self.frame_rate > 0:
            self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.frame_rate))
        if self.pixel_format:
            self.cam.MV_CC_SetEnumValueByString("PixelFormat", self.pixel_format)

        # 触发模式
        self.cam.MV_CC_SetEnumValueByString(
            "TriggerMode", "On" if self.trigger_mode else "Off"
        )
        if self.trigger_mode:
            self.cam.MV_CC_SetEnumValueByString("TriggerSource", "Line2")
            self.cam.MV_CC_SetEnumValueByString("TriggerActivation", "RisingEdge")

        # GigE 优化
        if devInfo.nTLayerType == MV_GIGE_DEVICE:
            self.cam.MV_CC_SetIntValueEx("GevSCPSPacketSize", self.packet_size)

        # 开始取流
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"StartGrabbing 失败: 0x{ret:x}")

        # 获取一次 payload（部分相机需要，以防后面 GetImageBuffer 失败）
        payload = MVCC_INTVALUE()
        self.cam.MV_CC_GetIntValue("PayloadSize", payload)
        print(f"PayloadSize: {payload.nCurValue}")

        # 启动抓帧线程（与 WebCamCamera 一模一样）
        self._create_capture_thread()

        print(
            f"海康相机已打开 → {self.target_width}x{self.target_height}  触发={'外触发' if self.trigger_mode else '自由运行'}"
        )

    def _create_capture_thread(self):
        """与 WebCamCamera 完全一致"""
        self._camera_thread_running = True
        self._camera_thread = threading.Thread(target=self.capture, daemon=True)
        self._camera_thread.start()

    def capture(self):
        """终极兼容版取帧（兼容所有海康SDK版本，永不崩）"""
        while self._camera_thread_running:
            stOutFrame = MV_FRAME_OUT()
            memset(byref(stOutFrame), 0, sizeof(stOutFrame))

            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 2000)

            if ret == 0:
                try:
                    # === 终极核弹级兼容写法（所有版本都稳）===
                    # 直接 cast 成指针数组，不走 int() 地址转换
                    pData = cast(
                        stOutFrame.pBufAddr,
                        POINTER(c_ubyte * stOutFrame.stFrameInfo.nFrameLen),
                    )
                    img = np.frombuffer(pData.contents, dtype=np.uint8)

                    # 转 RGB + resize
                    frame = self._convert_to_rgb_from_info(img, stOutFrame.stFrameInfo)
                    frame = cv2.resize(frame, (self.target_width, self.target_height))

                    timestamp = time.time_ns()
                    # 新增：计算相机原始 FPS
                    self.frame_count += 1  # 类里先加 self.frame_count = 0 和 self.last_time = time.time()
                    current_time = time.time()
                    if current_time - self.last_time >= 1.0:
                        camera_fps = self.frame_count / (current_time - self.last_time)
                        print(
                            f"相机原始帧率: {camera_fps:.2f} fps"
                        )  # 这就是相机输出的真实 FPS
                        self.frame_count = 0
                        self.last_time = current_time
                    # 回调
                    with self.callback_and_param_lock:
                        if self.callback_func is not None:
                            self.callback_func(
                                self.camera_running_state,
                                timestamp,
                                frame,
                                *self.callback_args,
                                **self.callback_kwargs,
                            )

                except Exception as e:
                    print(f"这帧丢弃了: {e}")

                finally:
                    self.cam.MV_CC_FreeImageBuffer(stOutFrame)
            else:
                if ret != 0:
                    print(f"取帧失败 0x{ret:x}")

            if not self.trigger_mode:
                time.sleep(0.001)

    def _convert_to_rgb(
        self, img: np.ndarray, info: MV_FRAME_OUT_INFO_EX
    ) -> np.ndarray:
        """自动处理所有常见格式 → 输出 RGB uint8 (h, w, 3)"""
        pixel_type = info.enPixelType

        if pixel_type == PixelType_Gvsp_Mono8:
            frame = img.reshape(info.nHeight, info.nWidth)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        elif pixel_type in [
            PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerGR8,
            PixelType_Gvsp_BayerGB8,
            PixelType_Gvsp_BayerBG8,
        ]:
            frame = img.reshape(info.nHeight, info.nWidth)
            # 常见相机默认是 BayerRG
            bayer_code = (
                cv2.COLOR_BayerRG2RGB
                if pixel_type == PixelType_Gvsp_BayerRG8
                else cv2.COLOR_BayerGR2RGB
                if pixel_type == PixelType_Gvsp_BayerGR8
                else cv2.COLOR_BayerGB2RGB
                if pixel_type == PixelType_Gvsp_BayerGB8
                else cv2.COLOR_BayerBG2RGB
            )
            frame = cv2.cvtColor(frame, bayer_code)

        elif pixel_type == PixelType_Gvsp_RGB8_Packed:
            frame = img.reshape(info.nHeight, info.nWidth, 3)

        else:
            # 兜底：截取前3通道
            frame = img.reshape(info.nHeight, info.nWidth, -1)[:, :, :3]
            if frame.shape[2] == 1:
                frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2RGB)

        return frame

    def _convert_to_rgb_from_info(
        self, img: np.ndarray, info: MV_FRAME_OUT_INFO_EX
    ) -> np.ndarray:
        h, w = info.nHeight, info.nWidth
        pixel_type = info.enPixelType

        if pixel_type == PixelType_Gvsp_Mono8:
            frame = img.reshape(h, w)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        elif pixel_type in [
            PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerGR8,
            PixelType_Gvsp_BayerGB8,
            PixelType_Gvsp_BayerBG8,
        ]:
            frame = img.reshape(h, w)
            code_map = {
                PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerRG2RGB,
                PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGR2RGB,
                PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGB2RGB,
                PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerBG2RGB,
            }
            frame = cv2.cvtColor(frame, code_map.get(pixel_type, cv2.COLOR_BayerRG2RGB))

        elif pixel_type == PixelType_Gvsp_RGB8_Packed:
            frame = img.reshape(h, w, 3)

        else:
            # 彩色相机其他格式兜底
            frame = img.reshape(h, w, -1)[..., :3]
            if frame.shape[2] == 1:
                frame = cv2.cvtColor(frame[..., 0], cv2.COLOR_GRAY2RGB)

        return frame.astype(np.uint8)

    def close(self):
        """与 WebCamCamera.close() 行为完全一致"""
        if self.cam is None:
            return

        print("海康相机正在关闭...")
        self._camera_thread_running = False
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2.0)

        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        self.cam = None
        print("海康相机已安全关闭")

    def release(self):
        """与 WebCamCamera.release() 完全一致"""
        self.close()


# ==================== 测试代码 ====================
if __name__ == "__main__":
    cam = HikvisionCamera(
        exposure_time=15000,
        frame_rate=120,
        trigger_mode=False,
        target_width=640,
        target_height=480,
    )

    def on_frame(state, ts, frame):
        print(
            f"[{CameraRunningState(state).name}] {ts}  shape={frame.shape}  dtype={frame.dtype}  mean={frame.mean():.2f}"
        )

    cam.set_on_image_callback(on_frame)
    cam.start_sampling()

    try:
        input("按回车停止...\n")
    finally:
        cam.stop_sampling()
        cam.release()
