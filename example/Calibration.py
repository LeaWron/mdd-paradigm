
from gazefollower import GazeFollower
from gazefollower.calibration import SVRCalibration
from gazefollower.misc import DefaultConfig
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog
import pygame  # 用于 UI 显示

#Initiate Calibration
# 配置
config = DefaultConfig()
config.cali_mode = 13  # 13 点校准

model_dir = Path(__file__).resolve().parent / "calib_models"
if not model_dir.exists():
    model_dir.mkdir(exist_ok=True)

# 采集被试信息并用作文件夹名
root = tk.Tk()
root.withdraw()  # 隐藏主窗口，仅显示对话框
try:
    subject_id = simpledialog.askstring("被试信息", "序号/ID:", parent=root) or "unknown"
    subject_name = simpledialog.askstring("被试信息", "姓名:", parent=root) or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
finally:
    root.destroy()

def _sanitize(s: str) -> str:
    # 仅保留字母数字、下划线与中划线，其他替换为下划线
    return "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in s.strip()) or "unknown"

subject_id_s = _sanitize(subject_id)
subject_name_s = _sanitize(subject_name)
default_folder = f"{subject_id_s}_{subject_name_s}_{timestamp}"

# 再次弹窗允许用户自定义文件夹名（默认带入建议）
root2 = tk.Tk()
root2.withdraw()
try:
    custom_folder = simpledialog.askstring("保存位置", "自定义文件夹名称:", initialvalue=default_folder, parent=root2) or default_folder
finally:
    root2.destroy()

folder_name = _sanitize(custom_folder)
target_dir = model_dir / folder_name
target_dir.mkdir(parents=True, exist_ok=True)

calibration = SVRCalibration(model_save_path=str(target_dir))
# 自定义校准类，指定保存路径（固定文件名 svr_x.xml / svr_y.xml）

# 初始化 GazeFollower
gf = GazeFollower(calibration=calibration, config=config)

# 初始化 pygame 用于显示校准界面（全屏）
pygame.init()
info = pygame.display.Info()
width = info.current_w
height = info.current_h
win = pygame.display.set_mode((width, height), pygame.FULLSCREEN)  # 调整为你的屏幕分辨率
#Start Calibration
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
pygame.quit()
#Save Calibrated Model

