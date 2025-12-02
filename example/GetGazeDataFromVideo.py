from gazefollower import GazeFollower
from gazefollower.calibration import SVRCalibration
from gazefollower.face_alignment import MediaPipeFaceAlignment
from gazefollower.gaze_estimator import MGazeNetGazeEstimator
from gazefollower.filter import HeuristicFilter
import cv2
import numpy as np
import csv
import time
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
model_save_path = current_dir.joinpath("calib_models")
print(f"模型路径: {model_save_path}")
if not model_save_path.exists():
    exception_msg = f"Calibration model directory does not exist: {model_save_path}. Please run the calibration script first."
    print(exception_msg)
    exit()
# 加载校准模型（指定相同路径）
calibration = SVRCalibration(model_save_path=str(model_save_path))
svm = cv2.ml.SVM_load(str(model_save_path.joinpath("svr_x.xml")))

# 获取并打印参数
print("SVM Type:", svm.getType())  # e.g., 103 for EPS_SVR
print("Kernel Type:", svm.getKernelType())  # e.g., 2 for RBF
print("Gamma:", svm.getGamma())
print("C:", svm.getC())
print("P (epsilon):", svm.getP())
# term_criteria = svm.getTermCriteria()
# print("Term Criteria - Iterations:", term_criteria.maxCount)
print("Var Count (features):", svm.getVarCount())
support_vectors = svm.getSupportVectors()
print("Support Vectors Total:", support_vectors.shape[0])  # rows
# 初始化 GazeFollower（无需相机，因为离线处理）
gf = GazeFollower(
    face_alignment=MediaPipeFaceAlignment(),
    gaze_estimator=MGazeNetGazeEstimator(),
    gaze_filter=HeuristicFilter(),
    calibration=calibration
)

# 确认加载
if gf.calibration.has_calibrated:
    print("SVR 模型已加载，可用于视频处理。")
    # 打印模型 var_count（诊断）
    print(f"模型 var_count: X={gf.calibration.svr_x.getVarCount()}, Y={gf.calibration.svr_y.getVarCount()}")
    print(1234567890)
else:
    print("模型加载失败，请检查路径。")
    exit()  # 或运行校准脚本

# 输入视频路径（心理学实验录制的人脸视频）
video_path=current_dir.joinpath("experiment_videos/Face_Test.mp4")
output_csv = current_dir.joinpath("output_eye_data/subject1_gaze.csv")


# 打开视频
cap = cv2.VideoCapture(str(video_path))



if not cap.isOpened():
    print("视频打开失败。")
    exit()

# 输出 CSV 头
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp', 'raw_x', 'raw_y', 'calibrated_x', 'calibrated_y', 'filtered_x', 'filtered_y'])

# 准备显示窗口与可选视频写出
cv2.namedWindow("Gaze Replay", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gaze Replay", 960, 540)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
annotated_video_path = current_dir.joinpath("output_eye_data/subject1_gaze_annotated.mp4")
os.makedirs(annotated_video_path.parent, exist_ok=True)
video_writer = None  # 如需保存带注释的视频，设置为 VideoWriter

def to_pixel(coords, w, h):
    if coords is None:
        return None
    x, y = float(coords[0]), float(coords[1])
    # 简单启发式：若值在[-2,2]之间，按归一化处理
    if -2.0 <= x <= 2.0 and -2.0 <= y <= 2.0:
        px = int(np.clip(x, 0.0, 1.0) * w)
        py = int(np.clip(y, 0.0, 1.0) * h)
    else:
        px = int(np.clip(x, 0, w - 1))
        py = int(np.clip(y, 0, h - 1))
    return px, py

def draw_point(img, pt, color, label=None):
    if pt is None:
        return
    cv2.circle(img, pt, 6, color, -1, lineType=cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 模拟 timestamp
    timestamp = time.time_ns()

    # 转换为 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 面部对齐
    face_info = gf.face_alignment.detect(timestamp, frame_rgb)

    if face_info.status and face_info.can_gaze_estimation:
        # 注视估计
        gaze_info = gf.gaze_estimator.detect(frame_rgb, face_info)
        feat = np.asarray(gaze_info.features)
        print(f"feat.shpe: {feat.shape}, feat.dtype: {feat.dtype}, feat.size = {feat.size}")
        if gaze_info.status:
            # 诊断：打印 features 信息（仅前几帧）
            if frame_count < 5:
                print(f"帧 {frame_count}: features shape {gaze_info.features.shape}, dtype {gaze_info.features.dtype}")

            # SVR 校准
            _, calibrated_coords = gf.calibration.predict(gaze_info.features, gaze_info.raw_gaze_coordinates)

            # 过滤
            filtered_coords = gf.gaze_filter.filter_values(calibrated_coords, timestamp)

            # 写入 CSV
            with open(output_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    timestamp,
                    gaze_info.raw_gaze_coordinates[0], gaze_info.raw_gaze_coordinates[1],
                    calibrated_coords[0], calibrated_coords[1],
                    filtered_coords[0], filtered_coords[1]
                ])

            # 绘制注视点到当前帧
            h, w = frame.shape[:2]
            raw_px = to_pixel(gaze_info.raw_gaze_coordinates, w, h)
            cal_px = to_pixel(calibrated_coords, w, h)
            fil_px = to_pixel(filtered_coords, w, h)
            vis = frame.copy()
            draw_point(vis, raw_px, (0, 165, 255), "raw")       # 橙色
            draw_point(vis, cal_px, (0, 255, 0), "calib")       # 绿色
            draw_point(vis, fil_px, (255, 0, 0), "filt")        # 蓝色
            # 图例
            cv2.rectangle(vis, (10, 10), (170, 80), (0, 0, 0), -1)
            cv2.putText(vis, "raw", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
            cv2.putText(vis, "calib", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(vis, "filt", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            cv2.imshow("Gaze Replay", vis)
            # 可选视频写出（如需，取消注释）
            # if video_writer is None:
            #     video_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (w, h))
            # video_writer.write(vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    frame_count += 1
    print(f"处理帧: {frame_count}")

# 释放
cap.release()
gf.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
print(f"处理完成，输出文件: {output_csv}")