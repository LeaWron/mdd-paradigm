import importlib
import logging
import multiprocessing
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
import psychopy
from omegaconf import DictConfig, OmegaConf
from psychopy import core, event, gui, visual  # noqa: E402

from psycho.camera import (  # noqa: E402
    close_camera,
    init_camera,
    init_record_thread,
    start_record,
    stop_record,
)
from psycho.utils import (  # noqa: E402
    PsychopyDisplaySelector,
    get_audio_devices,
    init_lsl,
    save_csv_data,
    select_monitor,
    send_marker,
    switch_keyboard_layout,
)

# 全局设置
PSYCHO_FONT = "Microsoft YaHei"
USE_CAMERA = False
USE_FNIRS = False


@dataclass
class Experiment:
    win: visual.Window
    clock: core.Clock
    lsl_outlet: None
    config: DictConfig
    logger: logging.Logger
    session_info: dict
    test: bool = False
    camera: None = None  # transfer camera handler


class Session:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.exps_dir = Path(cfg.exps_dir)
        self.experiments = []
        self.running = False
        self.win = None
        self.globalClock = core.Clock()
        self.trialClock = core.Clock()

        self.MONITOR_NAME = "subMonitor"

        self.camera = None

        self.continue_keys = ["space"]
        self.lsl_proc = None
        self.lsl_outlet = None

        self.before_duration = self.cfg.session.timing.before_wait
        self.after_rest_duration = self.cfg.session.timing.iei

        self.labrecorder_thread = threading.Thread(
            target=self._connect_labrecorder, daemon=True
        )

        event.globalKeys.add(
            key="escape", modifiers=["shift"], func=self.stop, name="quit"
        )
        # event.globalKeys.add(
        #     key="p", modifiers=["shift"], func=self.pause, name="pause"
        # )

    def setup_logger(self):
        base_dir = Path(self.cfg.output_dir) / datetime.now().strftime("%Y-%m-%d")
        base_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            filename=base_dir / (self.session_info["session_id"] + ".log"),
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(filename)s:%(lineno)d] - %(message)s"
            )
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("[%(levelname)s] - %(message)s"))

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG if self.cfg.debug else logging.INFO)
        self.logger.handlers = [file_handler, console_handler]

    def _connect_labrecorder(self):
        try:
            self.labrecorder_connection = socket.create_connection(
                (self.cfg.labrecorder.host, self.cfg.labrecorder.port)
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to LabRecorder: {e}")
            self.labrecorder_connection = None

    def setup_camera(self):
        self.camera = init_camera(
            save_dir=Path(self.cfg.output_dir) / "videos",
            file_name=f"{self.session_info['save_path']}_video.avi",
        )
        if self.camera is None:
            self.logger.error("Failed to initialize camera.")
            core.quit()

    def select_screen(self):
        selector = PsychopyDisplaySelector()
        self.selected_screen = selector.select_and_preview()

        self.params = selector.get_selected_screen_window_params()

    def select_monitor(self):
        self.select_screen()
        self.monitor = select_monitor(self.params, self.MONITOR_NAME, self.logger)

    def discover_experiments(self):
        files = list(self.exps_dir.glob("*.py"))
        exps = [
            exp.stem
            for exp in files
            if not exp.name.startswith("__init__")
            and exp.stem in self.cfg.session.default_order
        ]
        return exps

    def select_experiments_gui(self, exps):
        dlg = gui.Dlg(title="Select Experiments", screen=0)
        for exp in exps:
            dlg.addField(f"Run {exp}?", initial=True)
        ok = dlg.show()
        if not dlg.OK:
            core.quit()
        selected = [exp for exp, choice in zip(exps, ok.values()) if choice]
        return selected

    def sort_experiments(self, exp_list: list[str]):
        num_exps = len(exp_list)
        default_order_all = self.cfg.session.default_order
        default_order = [exp for exp in default_order_all if exp in exp_list]

        while True:
            dlg = gui.Dlg(title="选择实验顺序", screen=0)
            dlg.addText("请确保每个实验只出现一次，如果退出则会使用默认顺序")
            for i in range(0, num_exps):
                dlg.addField(
                    f"第 {i + 1} 个实验",
                    choices=default_order[i : i + 1]
                    + default_order[:i]
                    + default_order[i + 1 :],
                )

            ok_data = dlg.show()
            if not dlg.OK:
                break

            # 检查互斥
            if len(set(ok_data.values())) == num_exps:
                break
            else:
                gui.infoDlg("错误", "实验选择重复！请确保每个实验只出现一次。")

        # ok_data 顺序即为最终实验顺序
        order = list(ok_data.values()) if ok_data else default_order
        exp_list.sort(
            key=lambda x: order.index(x.lower()) if x.lower() in order else num_exps
        )
        return exp_list

    def add_experiments(self, exp_names):
        for name in exp_names:
            mod = importlib.import_module(f"psycho.exps.{name}")
            self.experiments.append((name, mod))

    def add_session_info(self):
        session_info = dict()
        dlg = gui.Dlg(title="Session Info", screen=0)
        # 序号
        dlg.addField(label="Session ID *", key="session_id", required=True)
        # 日期
        dlg.addFixedField(
            label="日期", initial=datetime.now().strftime("%Y-%m-%d"), key="date"
        )
        # 受试信息
        dlg.addField(label="受试信息 *", key="participant_id")
        # ..... 其他信息
        dlg.addField(
            label="数据保存路径",
            initial="",
            key="save_path",
            tip="csv 数据会保存到 data 目录下当前日期下, 这里需要指定一个前缀, 默认会是 session_id",
        )

        ok_data = dlg.show()
        if not dlg.OK:
            core.quit()
        if not ok_data["save_path"]:
            ok_data["save_path"] = ok_data["session_id"]
        session_info.update(ok_data)

        self.session_info = session_info
        if self.cfg.debug:
            print(session_info)

    def start(self, with_lsl=False):
        self.running = True

        if with_lsl:
            self.lsl_proc = multiprocessing.Process(target=self._lsl_recv)
            self.lsl_proc.start()

        if self.camera is not None:
            self.record_thread = init_record_thread(self.camera)
        try:
            self.lsl_outlet = init_lsl("ParadigmMarker")
            if USE_FNIRS:
                gui.infoDlg(title="请等待主试确认", prompt="请检查是否打开 fNIRS 录制")
            if self.labrecorder_connection is not None:
                # 文件名格式: {root}/{session_id}.xdf
                root = Path(self.cfg.output_dir) / self.session_info["date"]
                file_name_cmd = f"filename {{root:{root}}} {{template:%s.xdf}} {{session:{self.session_info['session_id']}}}\n"
                self.labrecorder_connection.sendall(file_name_cmd.encode("utf-8"))

                self.labrecorder_connection.sendall(b"update\n")
                time.sleep(0.5)

                self.labrecorder_connection.sendall(b"select all\n")
                # screen: 1 0 2
            time.sleep(1)
            self.win = visual.Window(
                monitor=self.monitor,
                screen=self.selected_screen,
                size=self.params["pix_size"] if not self.cfg.debug else (1600, 900),
                pos=(0, 0) if not self.cfg.debug else (200, 200),
                fullscr=True if not self.cfg.debug else False,
                allowGUI=False,
                color="grey",
                units="norm",
            )  # 全局窗口
            self.win.setMouseVisible(visibility=False)
            self.win.callOnFlip(event.clearEvents)

            initial_msg = visual.TextBox2(
                self.win,
                text="你即将开始本次会话, 准备好后按下<c=#51d237>空格键</c>开始",
                color="white",
                letterHeight=0.1,
                size=(2, None),
                alignment="center",
                font=PSYCHO_FONT,
            )
            initial_msg.draw()
            self.win.flip()

            while True:
                keys = event.waitKeys(modifiers=True)
                self.logger.debug(keys)
                if "space" == keys[0][0]:
                    break

            if self.labrecorder_connection is not None:
                self.labrecorder_connection.sendall(b"start\n")

            time.sleep(2)

            if self.camera is not None:
                start_record(self.camera, self.record_thread)
            self.session_start_time = self.globalClock.getTime()
            send_marker(self.lsl_outlet, "SESSION_START")

            self.win.flip()

            for i, (name, exp_module) in enumerate(self.experiments):
                if not self.running:
                    break
                start_msg = visual.TextBox2(
                    self.win,
                    text="准备进入"
                    + ("下" if i else "第")
                    + "一个实验, 按<c=#51d237>空格键</c>继续",
                    color="white",
                    letterHeight=0.1,
                    size=(2, None),
                    alignment="center",
                    font=PSYCHO_FONT,
                )
                start_msg.draw()
                self.win.flip()
                event.waitKeys(keyList=self.continue_keys)
                self.win.flip()

                # 多进程资源共享不了,直接退出(win 没送过去)
                exp_module.entry(
                    Experiment(
                        win=self.win,
                        clock=self.trialClock,
                        lsl_outlet=self.lsl_outlet,
                        config=self.cfg.exps[name],
                        logger=self.logger,
                        session_info=self.session_info,
                        test="test" in self.cfg and self.cfg.test,
                        camera=self.camera,
                    ),
                )

                end_msg = visual.TextBox2(
                    self.win,
                    text="该实验结束\n当你休息好后,按<c=#51d237>空格键</c>继续",
                    color="white",
                    letterHeight=0.1,
                    size=(2, None),
                    alignment="center",
                    font=PSYCHO_FONT,
                )
                end_msg.draw()
                self.win.flip()
                event.waitKeys(keyList=self.continue_keys)
                core.wait(0.3)
                self.win.flip()

        finally:
            self.stop()

    def _lsl_recv(self):
        from psycho.lsl_recv import main as recv_lsl

        recv_lsl()

    def stop(self):
        self.running = False
        if self.lsl_proc:
            self.lsl_proc.terminate()
        if self.win:
            visual.TextBox2(
                self.win,
                text="会话结束\n按<c=#51d237>空格键</c>退出",
                color="white",
                letterHeight=0.1,
                size=(2, None),
                alignment="center",
                font=PSYCHO_FONT,
            ).draw()
            self.win.flip()
            event.waitKeys(keyList=self.continue_keys)
            self.win.close()

        send_marker(self.lsl_outlet, "SESSION_END")
        if self.camera is not None:
            stop_record(self.camera, self.record_thread)
            close_camera(self.camera)
        if self.labrecorder_connection is not None:
            self.labrecorder_connection.sendall(b"stop\n")

        session_end_time = self.globalClock.getTime()

        # 准备session信息数据
        session_data = {
            "session_id": [self.session_info.get("session_id", "unknown")],
            "participant_id": [self.session_info.get("participant_id", "unknown")],
            "date": [self.session_info.get("date", "unknown")],
            "session_start_time": [self.session_start_time],
            "session_end_time": [session_end_time],
            "session_duration_seconds": [(session_end_time - self.session_start_time)],
            "experiments_count": [len(self.experiments)],
            "experiment_names": [",".join([exp[0] for exp in self.experiments])],
        }

        # 使用save_csv_data函数保存session信息
        file_name = f"{self.session_info.get('save_path', 'session_info')}_session_info"
        try:
            save_csv_data(session_data, file_name)
            self.logger.info(f"Session info saved to CSV: {file_name}")
        except Exception as e:
            self.logger.error(f"Failed to save session info to CSV: {e}")

        # core.quit()

    def pause(self):
        """暂停会话, 功能未实现"""
        pause_start = self.trialClock.getTime()  # 记录暂停开始时间（系统时间）

        # 只用管理 win 和 clock 即可

        paused = True

        send_marker(self.lsl_outlet, "SESSION_PAUSE")
        while paused:
            gui.infoDlg(
                title="会话暂停",
                prompt="会话已暂停，按 r 恢复",
            )
            keys = event.getKeys()
            if "r" in keys:  # 按 r 恢复
                paused = False
        send_marker(self.lsl_outlet, "SESSION_RESUME")

        # === 校正时钟 ===
        self.trialClock.reset(-pause_start)

        # pause_end = core.getTime()
        # pause_duration = pause_end - pause_start
        # self.globalClock.addTime(-pause_duration)
        # self.trialClock.addTime(-pause_duration)


@hydra.main(version_base=None, config_path="conf", config_name="pilot")
def run_session(cfg: DictConfig):
    # 切换到英文输入法
    switch_keyboard_layout()
    # 初始化音视频设备
    init_external_device()

    OmegaConf.resolve(cfg)
    if cfg.debug:
        with open("temp_debug.yaml", "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg))
            print("temp_debug.yaml 已生成")
    session = Session(cfg)
    session.add_session_info()
    session.setup_logger()

    session.labrecorder_thread.start()
    if USE_CAMERA:
        session.setup_camera()
    session.select_monitor()

    exps = session.discover_experiments()
    selected = session.select_experiments_gui(exps)
    sort = session.sort_experiments(selected)
    session.add_experiments(sort)
    session.start(with_lsl=False)


def init_external_device():
    sound_devices = get_audio_devices()
    # gui 选择
    select_device = gui.Dlg("外部设备选择")

    select_device.addField(
        label="请选择扬声器",
        choices=sound_devices,
        key="audio_device",
        tip="默认为第一项",
    )
    select_device.addField(label="是否使用视频设备", initial=False, key="use_video")
    select_device.addField(label="是否使用 fNIRS 设备", initial=False, key="use_fnirs")
    ok_data = select_device.show()
    if select_device.OK:
        psychopy.prefs.hardware["audioDevice"] = ok_data["audio_device"]
        if ok_data["use_video"]:
            global USE_CAMERA
            USE_CAMERA = True
        if ok_data["use_fnirs"]:
            global USE_FNIRS
            USE_FNIRS = True


if __name__ == "__main__":
    run_session()
