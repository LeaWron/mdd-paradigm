import importlib
import logging
import multiprocessing
import socket
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
import psychopy
from omegaconf import DictConfig, OmegaConf

from psycho.camera import (
    close_camera,
    init_camera,
    init_record_thread,
    start_record,
    stopRecord,
)
from psycho.utils import (
    get_audio_devices,
    init_lsl,
    send_marker,
    switch_keyboard_layout,
)

# 全局设置
psychopy.prefs.general["defaultTextFont"] = "Arial"
psychopy.prefs.general["defaultTextSize"] = 0.05
psychopy.prefs.general["defaultTextColor"] = "white"
from psychopy import core, event, gui, visual  # noqa: E402

USE_CAMERA = False


@dataclass
class Experiment:
    win: visual.Window
    clock: core.Clock
    lsl_outlet: None
    config: DictConfig
    logger: logging.Logger
    session_info: dict
    test: bool = False


class Session:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.exps_dir = Path(cfg.exps_dir)
        self.experiments = []
        self.running = False
        self.win = None
        self.globalClock = core.Clock()
        self.trialClock = core.Clock()

        self.camera = None
        if USE_CAMERA:
            self.camera = init_camera()
            if self.camera is None:
                self.logger.error("Failed to initialize camera.")
                core.quit()

        self.continue_keys = ["space"]
        self.lsl_proc = None
        self.lsl_outlet = None

        self.before_duration = self.cfg.session.timing.before_wait
        self.after_rest_duration = self.cfg.session.timing.iei

        self.labrecorder_connection = None
        if "labrecorder" in self.cfg and ("test" not in self.cfg or self.cfg.test):
            self.labrecorder_connection = socket.create_connection(
                (self.cfg.labrecorder.host, self.cfg.labrecorder.port)
            )

        event.globalKeys.add(
            key="escape", modifiers=["shift"], func=self.stop, name="quit"
        )
        event.globalKeys.add(
            key="p", modifiers=["shift"], func=self.pause, name="pause"
        )

        # 初始化 logger
        self._setup_logger()

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.DEBUG if self.cfg.debug else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

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
        dlg.addField(label="受试信息", key="participant_id")
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
        # screen: 1 0 2
        self.win = visual.Window(
            monitor="testMonitor",
            screen=1,
            pos=(0, 0),
            fullscr=True,
            color="grey",
            units="norm",
        )  # 全局窗口
        self.win.callOnFlip(event.clearEvents)
        if with_lsl:
            self.lsl_proc = multiprocessing.Process(target=self._lsl_recv)
            self.lsl_proc.start()

        if self.camera is not None:
            self.record_thread = init_record_thread(self.camera)
        try:
            self.lsl_outlet = init_lsl("ParadigmMarker")
            if self.labrecorder_connection is not None:
                # 文件名格式: {root}/{session_id}.xdf
                root = Path(self.cfg.output_dir) / self.session_info["date"]
                file_name_cmd = f"filename {{root:{root}}} {{template:%s}} {{session:{self.session_info['session_id']}}}.xdf"
                self.labrecorder_connection.sendall(file_name_cmd.encode("utf-8"))

                self.labrecorder_connection.sendall(b"update\n")
                self.labrecorder_connection.sendall(b"select all\n")

            initial_msg = visual.TextStim(
                self.win,
                text="你即将开始本次会话, 准备好后按下空格键开始",
                color="white",
                height=0.1,
                wrapWidth=2,
            )
            initial_msg.draw()
            self.win.flip()

            while True:
                keys = event.waitKeys(modifiers=True)
                print(keys)
                if "space" == keys[0][0]:
                    break

            if self.labrecorder_connection is not None:
                self.labrecorder_connection.sendall(b"start\n")
            if self.camera is not None:
                start_record(self.camera, self.record_thread)
            send_marker(self.lsl_outlet, "SESSION_START")

            self.win.flip()

            for name, exp_module in self.experiments:
                if not self.running:
                    break
                start_msg = visual.TextStim(
                    self.win,
                    text="准备进入实验, 按空格键继续",
                    color="white",
                    height=0.1,
                    wrapWidth=2,
                )
                start_msg.draw()
                self.win.flip()
                event.waitKeys(self.before_duration, keyList=self.continue_keys)
                core.wait(0.3)
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
                    ),
                )

                end_msg = visual.TextStim(
                    self.win,
                    text=f"该实验结束, 你有 {self.after_rest_duration} 秒休息时间\n你可以按空格键直接进入下一个实验",
                    color="white",
                    height=0.1,
                    wrapWidth=2,
                )
                end_msg.draw()
                self.win.flip()
                event.waitKeys(self.after_rest_duration, keyList=self.continue_keys)
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
            self.win.close()

        send_marker(self.lsl_outlet, "SESSION_END")
        if self.labrecorder_connection is not None:
            self.labrecorder_connection.sendall(b"stop\n")
        if self.camera is not None:
            stopRecord(self.camera, self.record_thread)
            close_camera(self.camera)
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
    init_audio_video_device()

    OmegaConf.resolve(cfg)
    if cfg.debug:
        print(cfg)
    session = Session(cfg)
    session.add_session_info()

    exps = session.discover_experiments()
    selected = session.select_experiments_gui(exps)
    sort = session.sort_experiments(selected)
    session.add_experiments(sort)
    session.start(with_lsl=False)


def init_audio_video_device():
    sound_devices = get_audio_devices()
    # gui 选择
    select_audio_vedio = gui.Dlg("音视频设备选择")
    select_audio_vedio.addField(
        label="请选择", choices=sound_devices, key="audio_device", tip="默认为第一项"
    )
    select_audio_vedio.addField(
        label="是否使用视频设备", initial=False, key="use_video"
    )
    ok_data = select_audio_vedio.show()
    if select_audio_vedio.OK:
        psychopy.prefs.hardware["audioDevice"] = ok_data["audio_device"]
        if ok_data["use_video"]:
            global USE_CAMERA
            USE_CAMERA = True


if __name__ == "__main__":
    run_session()
