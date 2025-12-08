import os
import sys
import threading
import time
from pathlib import Path

import serial
from pylsl import StreamOutlet

from psycho.utils import init_lsl, send_marker, setup_default_logger

HAS_MV_CAMERA = os.getenv("MVCAM_COMMON_RUNENV") is not None

if HAS_MV_CAMERA:
    sys.path.append(os.getenv("MVCAM_COMMON_RUNENV") + "/Samples/Python/MvImport")
    from MvCameraControl_class import *  # noqa
    from CameraParams_header import *


EXPOSURE_TIME = 10_000.0
FRAME_RATE = 90.0
GAIN = 12.5

# 全局变量控制录像线程
g_bExit = False
ser: serial.Serial = None
lsl_outlet: StreamOutlet = None
logger = setup_default_logger()


# 为线程定义一个函数
def work_thread(cam, pData, nDataSize):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    stInputFrameInfo = MV_CC_INPUT_FRAME_INFO()
    memset(byref(stInputFrameInfo), 0, sizeof(MV_CC_INPUT_FRAME_INFO))
    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret:
            cur_time_stamp = int(round(time.time() * 1000))
            logger.debug(
                f"get one frame: Width[{stOutFrame.stFrameInfo.nWidth}], Height[{stOutFrame.stFrameInfo.nHeight}], nFrameNum[{stOutFrame.stFrameInfo.nFrameNum}] nFrameLen[{stOutFrame.stFrameInfo.nFrameLen}] fram_time_stamp[{stOutFrame.stFrameInfo.nHostTimeStamp}] host_time_stamp[{cur_time_stamp}]"
            )
            stInputFrameInfo.pData = cast(stOutFrame.pBufAddr, POINTER(c_ubyte))
            stInputFrameInfo.nDataLen = stOutFrame.stFrameInfo.nFrameLen
            # ch:输入一帧数据到录像接口|en:Input a frame of data to the video interface
            ret = cam.MV_CC_InputOneFrame(stInputFrameInfo)
            send_marker(
                lsl_outlet,
                {
                    "frame_id": stOutFrame.stFrameInfo.nFrameNum,
                    "host_timestamp": cur_time_stamp,
                },
            )
            if ret != 0:
                logger.error(f"input one frame fail! nRet [0x{ret:08x}]")
            nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
        else:
            logger.debug(f"no data[0x{ret:08x}]")
        if g_bExit is True:
            break


def send_lowlevel_trigger():
    # 低电平触发
    ser.rts = True


def send_highlevel_trigger():
    # 高电平触发
    ser.rts = False


def init_camera(save_dir: Path = None, file_name: str = None):
    global lsl_outlet
    lsl_outlet = init_lsl("CameraMaker")

    # 初始化SDK
    ret = MvCamera.MV_CC_Initialize()
    if ret != 0:
        logger.error(f"Initialize SDK failed! ret [0x{ret:08x}]")
        return None

    # 枚举设备
    device_list = MV_CC_DEVICE_INFO_LIST()
    tlayer_type = MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
    if ret != 0:
        logger.error(f"Enum devices failed! ret [0x{ret:08x}]")
        return None

    if device_list.nDeviceNum == 0:
        logger.error("No camera found!")
        return None

    for i in range(0, device_list.nDeviceNum):
        mvcc_dev_info = cast(
            device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)
        ).contents
        if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            logger.info(f"\nu3v device: [{i}]")
            strModeName = "".join(
                [
                    chr(c)
                    for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName
                    if c != 0
                ]
            )
            logger.info(f"device model name: [{strModeName}]")

            strSerialNumber = "".join(
                [
                    chr(c)
                    for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber
                    if c != 0
                ]
            )
            logger.info(f"user serial number: [{strSerialNumber}]")
    # 选择第一个设备
    # nConnectionNum = input("please input the number of the device to connect:")
    # 默认选择第一个设备
    n_connection_num = "0"
    if int(n_connection_num) >= device_list.nDeviceNum:
        logger.error("intput error!")
        return None
    # 选择设备
    st_device_list = cast(
        device_list.pDeviceInfo[int(n_connection_num)], POINTER(MV_CC_DEVICE_INFO)
    ).contents

    # 创建相机实例
    cam = MvCamera()
    # 创建句柄
    ret = cam.MV_CC_CreateHandle(st_device_list)
    if ret != 0:
        logger.error(f"Create handle failed! ret [0x{ret:08x}]")
        return None

    # 打开设备
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        logger.error(f"Open device failed! ret [0x{ret:08x}]")
        logger.error(f"Open device failed! ret [0x{ret:08x}]")
        cam.MV_CC_DestroyHandle()
        return None

    # ===================== 配置增益等 ======================
    ret = cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
    if ret != 0:
        logger.error(f"Set exposure auto: {ret}")
        return None
    ret = cam.MV_CC_SetFloatValue("ExposureTime", EXPOSURE_TIME)
    if ret != 0:
        logger.error(f"Set exposure time: {ret}")
        return None
    ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", FRAME_RATE)
    if ret != 0:
        logger.error(f"Set acquisition frame rate: {ret}")
        return None
    ret = cam.MV_CC_SetEnumValueByString("GainAuto", "Off")
    if ret != 0:
        logger.error(f"Set gain auto: {ret}")
        return None
    ret = cam.MV_CC_SetFloatValue("Gain", GAIN)
    if ret != 0:
        logger.error(f"Set gain: {ret}")
        return None
    ret = cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 1) # 自动白平衡
    if ret != 0:
        logger.error(f"Set balance white auto: {ret}")
        return None
    ret = cam.MV_CC_SetBoolValue("ReverseX", True)
    if ret != 0:
        logger.error(f"Set ReverseX failed! ret [0x{ret:08x}]")
        return None
    # ===================== 配置硬件触发 =====================

    # 设置触发模式为 on (外触发模式)
    ret = cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")  # 1=On
    if ret != 0:
        logger.error(f"Set TriggerMode failed! ret [0x{ret:08x}]")
        return None
    # # 设置触发源为Line2
    # ret = cam.MV_CC_SetEnumValueByString("TriggerSource", "Line2")  # 2=Line2
    # if ret != 0:
    #     logger.error(f"Set TriggerSource failed! ret [0x{ret:08x}]")
    #     return None
    # # 设置触发沿为 LevelLow
    # ret = cam.MV_CC_SetEnumValueByString("TriggerActivation", "LevelLow")  # 0=LevelLow
    # if ret != 0:
    #     logger.error(f"Set TriggerActivation failed! ret [0x{ret:08x}]")
    #     return None
    #
    # # 设置触发延迟
    # ret = cam.MV_CC_SetFloatValue("TriggerDelay", 0)
    # if ret != 0:
    #     logger.error(f"Set TriggerDelay failed! ret [0x{ret:08x}]")
    #     return None
    #
    # # 切换 LineSelector 为 Line2
    # ret = cam.MV_CC_SetEnumValueByString("LineSelector", "Line2")
    # if ret != 0:
    #     logger.error(f"Set LineSelector failed! ret [0x{ret:08x}]")
    #     return None
    # # 设置Line2 滤波时间(us)、误触发可适当加大
    # ret = cam.MV_CC_SetIntValueEx("LineDebouncerTime", 50)
    # if ret != 0:
    #     logger.error(f"Set LineDebouncerTime failed! ret [0x{ret:08x}]")
    #     return None

    # ======================================================

    # 获取相机参数
    st_param = MVCC_INTVALUE()
    memset(byref(st_param), 0, sizeof(st_param))
    ret = cam.MV_CC_GetIntValue("Width", st_param)
    if ret != 0:
        logger.error(f"Get width failed! ret [0x{ret:08x}]")
        return None
    n_width = st_param.nCurValue

    ret = cam.MV_CC_GetIntValue("Height", st_param)
    if ret != 0:
        logger.error(f"Get height failed! ret [0x{ret:08x}]")
        return None
    n_height = st_param.nCurValue

    st_enum_value = MVCC_ENUMVALUE()
    memset(byref(st_enum_value), 0, sizeof(st_enum_value))
    ret = cam.MV_CC_GetEnumValue("PixelFormat", st_enum_value)
    if ret != 0:
        logger.error(f"Get pixel format failed! ret [0x{ret:08x}]")
        return None
    en_pixel_type = MvGvspPixelType(st_enum_value.nCurValue)

    st_float_value = MVCC_FLOATVALUE()
    memset(byref(st_float_value), 0, sizeof(st_float_value))
    ret = cam.MV_CC_GetFloatValue("ResultingFrameRate", st_float_value)
    if ret != 0:
        logger.error(f"Get frame rate failed! ret [0x{ret:08x}]")
        return None
    f_frame_rate = st_float_value.fCurValue

    # 设置录像参数
    record_param = MV_CC_RECORD_PARAM()
    memset(byref(record_param), 0, sizeof(record_param))
    record_param.enPixelType = en_pixel_type
    record_param.nWidth = n_width
    record_param.nHeight = n_height
    record_param.fFrameRate = f_frame_rate
    # [ ]: 码率的选择需要测试
    record_param.nBitRate = 4096 * 1 # 码率(kbps) 4 Mbps * K
    record_param.enRecordFmtType = MV_FormatType_AVI  # 录像格式为AVI

    if save_dir is None:
        save_dir = Path.cwd()
    if file_name is None:
        file_name = "video.avi"
    save_path = save_dir / file_name
    record_param.strFilePath = save_path.as_posix().encode("utf-8")

    # 确保目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO create HikrobotCamera entity, with cam:MvCamera, then create thread in this class and capture videos

    # hikvision_camera=HikvisionCamera(camera_handler=cam) # 录像,取流均在这一个实例中进行
    # 开始录像
    ret = cam.MV_CC_StartRecord(record_param)
    if ret != 0:
        logger.error(f"Start record failed! ret [0x{ret:08x}]")
        return None
    # 开始取流
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        logger.error(f"Start grabbing failed! ret [0x{ret:08x}]")
        return None

    # send_lowlevel_trigger()
    return cam


def init_record_thread(cam):
    # 创建录像线程
    global g_bExit
    g_bExit = False
    record_thread = threading.Thread(target=work_thread, args=(cam, None, None))
    return record_thread


def start_record(cam, record_thread: threading.Thread):
    record_thread.start()
    send_marker(lsl_outlet, "StartRecording")


def stop_record(cam, record_thread: threading.Thread):
    global g_bExit
    # send_highlevel_trigger()
    # 停止录像
    g_bExit = True
    record_thread.join()
    send_marker(lsl_outlet, "StopRecording")

    time.sleep(0.5)

    ret = cam.MV_CC_StopRecord()
    if ret != 0:
        logger.error(f"Stop record failed! ret [0x{ret:08x}]")

    # 停止取流
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        logger.error(f"Stop grabbing failed! ret [0x{ret:08x}]")

    time.sleep(0.5)


def close_camera(cam):
    # 关闭设备
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()
    logger.info("Camera resources released.")


def main():
    cam = init_camera()
    if cam is not None:
        record_thread = init_record_thread(cam)
        start_record(cam, record_thread)
        input("Press any key to stop recording...")
        stop_record(cam, record_thread)
    else:
        MvCamera.MV_CC_Finalize()
    close_camera()


if __name__ == "__main__":
    main()
