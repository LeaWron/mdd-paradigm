import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # Adds GazeFollower-main/ to path
# [ ] UI 不重复使用, session info 不重复收集
import tkinter as tk
from datetime import datetime
from tkinter import simpledialog
from typing import Any

from gazefollower.calibration import SVRCalibration
from gazefollower.misc import DefaultConfig
from psychopy import core, visual  # 用于 UI 显示

from gazefollower import GazeFollower

from .HikrobotCamera import HikvisionCamera


def eyetracking_calibration(
    win: visual.window = None,
    formal: bool = False,
    info: dict = None,
    camera: Any | None = None,
):
    ret = 0
    # Initiate Calibration
    # 配置
    config = DefaultConfig()
    config.cali_mode = 13  # 13 点校准

    model_dir = Path(__file__).resolve().parent.parent.parent / "data" / "calib_models"
    if not model_dir.exists():
        model_dir.mkdir(exist_ok=True)

    # 采集被试信息并用作文件夹名
    target_dir = model_dir / info["session_id"]
    target_dir.mkdir(parents=True, exist_ok=True)

    calibration = SVRCalibration(model_save_path=str(target_dir))
    # 自定义校准类，指定保存路径（固定文件名 svr_x.xml / svr_y.xml）
    mycamera = camera
    print(f"camera is f{camera}")
    # 初始化 GazeFollower
    if mycamera is None:
        mycamera = HikvisionCamera()
    else:
        mycamera = HikvisionCamera(camera_handler=mycamera, formal=True)
    gf = GazeFollower(camera=mycamera, calibration=calibration, config=config)

    # 初始化 PsychoPy 用于显示校准界面（全屏）
    if win is None:
        win = visual.Window(
            fullscr=True, color="white", units="pix"
        )  # 全屏窗口，背景白色
    else:
        win = win
    # Start Calibration
    # 预览相机（可选，确保相机正常）
    gf.preview(win=win)

    # 运行校准（会自动训练并保存模型）
    gf.calibrate(win=win)

    # 确认保存（固定文件名）
    if gf.calibration.has_calibrated and gf.calibration.save_model():
        print("SVR 模型已保存到自定义文件夹:")
        print(f" - {target_dir / 'svr_x.xml'}")
        print(f" - {target_dir / 'svr_y.xml'}")
    else:
        print("校准失败，请检查日志。")

    # 释放资源
    gf.release()
    if formal is False:
        ret = 1
        win.close()
        core.quit()
    else:
        ret = 1
    # Save Calibrated Model
    return ret


if __name__ == "__main__":
    eyetracking_calibration()
