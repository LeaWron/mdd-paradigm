import ctypes
import os
import sys
import threading
from pathlib import Path
import serial

sys.path.append(os.getenv("MVCAM_COMMON_RUNENV") + "/Samples/Python/MvImport")
from MvCameraControl_class import *  # noqa
from CameraParams_header import *

# 全局变量控制录像线程
g_bExit = False
ser: serial.Serial = None


# 为线程定义一个函数
def work_thread(cam, pData, nDataSize):
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))

    stInputFrameInfo = MV_CC_INPUT_FRAME_INFO()
    memset(byref(stInputFrameInfo), 0, sizeof(MV_CC_INPUT_FRAME_INFO))
    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret:
            print(
                "get one frame: Width[%d], Height[%d], nFrameNum[%d]"
                % (
                    stOutFrame.stFrameInfo.nWidth,
                    stOutFrame.stFrameInfo.nHeight,
                    stOutFrame.stFrameInfo.nFrameNum,
                )
            )
            stInputFrameInfo.pData = cast(stOutFrame.pBufAddr, POINTER(c_ubyte))
            stInputFrameInfo.nDataLen = stOutFrame.stFrameInfo.nFrameLen
            # ch:输入一帧数据到录像接口|en:Input a frame of data to the video interface
            ret = cam.MV_CC_InputOneFrame(stInputFrameInfo)
            if ret != 0:
                print("input one frame fail! nRet [Ox%x]" % ret)
            nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
        else:
            print("no data[0x%x]" % ret)
            if g_bExit == True:
                break


def send_lowlevel_trigger():
    # 低电平触发
    ser.rts = True


def send_highlevel_trigger():
    # 高电平触发
    ser.rts = False


def init_camera():
    global ser
    ser = serial.Serial(port="COM3", baudrate=9600, timeout=1)


def startRecord(save_dir: Path = None, file_name: str = None):
    # 初始化SDK
    ret = MvCamera.MV_CC_Initialize()
    if ret != 0:
        print("Initialize SDK failed! ret = 0x%x" % ret)
        return None, None

    # 枚举设备
    device_list = MV_CC_DEVICE_INFO_LIST()
    tlayer_type = MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
    if ret != 0:
        print("Enum devices failed! ret = 0x%x" % ret)
        return None, None

    if device_list.nDeviceNum == 0:
        print("No camera found!")
        return None, None

    for i in range(0, device_list.nDeviceNum):
        mvcc_dev_info = cast(
            device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)
        ).contents
        if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = "".join(
                [
                    chr(c)
                    for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName
                    if c != 0
                ]
            )
            print("device model name: %s" % strModeName)

            strSerialNumber = "".join(
                [
                    chr(c)
                    for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber
                    if c != 0
                ]
            )
            print("user serial number: %s" % strSerialNumber)
    # 选择第一个设备
    nConnectionNum = input("please input the number of the device to connect:")

    if int(nConnectionNum) >= device_list.nDeviceNum:
        print("intput error!")
        return None, None
    # 选择设备
    st_device_list = cast(
        device_list.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)
    ).contents

    # 创建相机实例
    cam = MvCamera()
    # 创建句柄
    ret = cam.MV_CC_CreateHandle(st_device_list)
    if ret != 0:
        print("Create handle failed! ret = 0x%x" % ret)
        return None, None

    # 打开设备
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("Open device failed! ret = 0x%x" % ret)
        cam.MV_CC_DestroyHandle()
        return None, None

    # ===================== 配置硬件触发 =====================
    # 设置触发模式为 on (外触发模式)
    ret = cam.MV_CC_SetEnumValueByString("TriggerMode", "On")  # 1=On
    if ret != 0:
        print("Set TriggerMode failed! ret = 0x%x" % ret)
        return None, None
    # 设置触发源为Line2
    ret = cam.MV_CC_SetEnumValueByString("TriggerSource", "Line2")  # 2=Line2
    if ret != 0:
        print("Set TriggerSource failed! ret = 0x%x" % ret)
        return None, None
    # 设置触发沿为 LevelLow
    ret = cam.MV_CC_SetEnumValueByString("TriggerActivation", "LevelLow")  # 0=LevelLow
    if ret != 0:
        print("Set TriggerActivation failed! ret = 0x%x" % ret)
        return None, None

    # 设置触发延迟
    ret = cam.MV_CC_SetFloatValue("TriggerDelay", 0)
    if ret != 0:
        print("Set TriggerDelay failed! ret = 0x%x" % ret)
        return None, None

    # 切换 LineSelector 为 Line2
    ret = cam.MV_CC_SetEnumValueByString("LineSelector", "Line2")
    if ret != 0:
        print("Set LineSelector failed! ret = 0x%x" % ret)
        return None, None
    # 设置Line2 滤波时间(us)、误触发可适当加大
    ret = cam.MV_CC_SetIntValueEx("LineDebouncerTime", 50)
    if ret != 0:
        print("Set LineDebouncerTime failed! ret = 0x%x" % ret)
        return None, None

    # ======================================================

    # 获取相机参数
    st_param = MVCC_INTVALUE()
    memset(byref(st_param), 0, sizeof(st_param))
    ret = cam.MV_CC_GetIntValue("Width", st_param)
    if ret != 0:
        print("Get width failed! ret = 0x%x" % ret)
    n_width = st_param.nCurValue

    ret = cam.MV_CC_GetIntValue("Height", st_param)
    if ret != 0:
        print("Get height failed! ret = 0x%x" % ret)
    n_height = st_param.nCurValue

    st_enum_value = MVCC_ENUMVALUE()
    memset(byref(st_enum_value), 0, sizeof(st_enum_value))
    ret = cam.MV_CC_GetEnumValue("PixelFormat", st_enum_value)
    if ret != 0:
        print("Get pixel format failed! ret = 0x%x" % ret)
    en_pixel_type = st_enum_value.nCurValue

    st_float_value = MVCC_FLOATVALUE()
    memset(byref(st_float_value), 0, sizeof(st_float_value))
    ret = cam.MV_CC_GetFloatValue("ResultingFrameRate", st_float_value)
    if ret != 0:
        print("Get frame rate failed! ret = 0x%x" % ret)
    f_frame_rate = st_float_value.fCurValue

    # 设置录像参数
    record_param = MV_CC_RECORD_PARAM()
    memset(byref(record_param), 0, sizeof(record_param))
    record_param.enPixelType = en_pixel_type
    record_param.nWidth = n_width
    record_param.nHeight = n_height
    record_param.fFrameRate = f_frame_rate
    record_param.nBitRate = 4096  # 码率(kbps)
    record_param.enRecordFmtType = MV_FormatType_AVI  # 录像格式为AVI

    # 用户指定的保存路径
    if save_dir is None:
        save_dir = Path.cwd()
    if file_name is None:
        file_name = "video.avi"
    save_path = save_dir / file_name  # 修改为您需要的路径

    # 确保目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 开始录像
    ret = cam.MV_CC_StartRecord(record_param)
    if ret != 0:
        print("Start record failed! ret = 0x%x" % ret)
        return None, None
    # 开始取流
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("Start grabbing failed! ret = 0x%x" % ret)
        return None, None

    # send_lowlevel_trigger()
    # 创建并启动录像线程
    global g_bExit
    g_bExit = False
    record_thread = threading.Thread(target=work_thread, args=(cam, None, None))
    record_thread.start()
    return cam, record_thread


def stopRecord(cam, record_thread):
    global g_bExit
    send_highlevel_trigger()
    # 停止录像
    g_bExit = True
    record_thread.join()

    ret = cam.MV_CC_StopRecord()
    if ret != 0:
        print("Stop record failed! ret = 0x%x" % ret)

    # 停止取流
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("Stop grabbing failed! ret = 0x%x" % ret)

    # 关闭设备
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    MvCamera.MV_CC_Finalize()
    print("Camera resources released.")


def close_camera():
    global ser
    ser.close()


def main():
    init_camera()
    cam, record_thread = startRecord()
    if cam is not None:
        stopRecord(cam, record_thread)
    else:
        MvCamera.MV_CC_Finalize()
    close_camera()


if __name__ == "__main__":
    main()
