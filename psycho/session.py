# session.py
import importlib
import multiprocessing
from pathlib import Path

from psychopy import core, event, gui, visual

from psycho.utils import switch_keyboard_layout


class Session:
    def __init__(self, exps_dir="./psycho/exps"):
        self.exps_dir = Path(exps_dir)
        self.experiments = []
        self.running = False
        self.win = None
        self.globalClock = core.Clock()
        self.trialClock = core.Clock()

        self.continue_keys = ["space"]
        self.lsl_proc = None
        event.globalKeys.add(key="escape", modifiers=["shift"], func=self.stop, name="quit")
        event.globalKeys.add(key="p", modifiers=["shift"], func=self.pause, name="pause")

    def discover_experiments(self):
        files = list(self.exps_dir.glob("*.py"))
        exps = [exp.stem for exp in files if not exp.name.startswith("__init__")]
        return exps

    def select_experiments_gui(self, exps):
        dlg = gui.Dlg(title="Select Experiments")
        for exp in exps:
            dlg.addField(f"Run {exp}?", initial=True)
        ok = dlg.show()
        if not dlg.OK:
            core.quit()
        selected = [exp for exp, choice in zip(exps, ok.values()) if choice]
        return selected

    def sort_experiments(self, exp_list: list[str]):
        num_exps = len(exp_list)
        default_order_all = ["gng", "nback", "diat", "emotion_stim"]
        default_order = [exp for exp in default_order_all if exp in exp_list]

        while True:
            dlg = gui.Dlg(title="选择实验顺序")
            dlg.addText("请确保每个实验只出现一次，如果退出则会使用默认顺序")
            for i in range(0, num_exps):
                dlg.addField(f"第 {i + 1} 个实验", choices=default_order[i : i + 1] + default_order[:i] + default_order[i + 1 :])

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
        exp_list.sort(key=lambda x: order.index(x.lower()) if x.lower() in order else num_exps)
        return exp_list

    def add_experiments(self, exp_names):
        for name in exp_names:
            mod = importlib.import_module(f"psycho.exps.{name}")
            self.experiments.append(mod)

    def start(self, with_lsl=False):
        self.running = True
        self.win = visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")  # 全局窗口
        if with_lsl:
            self.lsl_proc = multiprocessing.Process(target=self._lsl_recv)
            self.lsl_proc.start()

        try:
            for exp in self.experiments:
                if not self.running:
                    break
                start_msg = visual.TextStim(
                    self.win,
                    text="准备进入实验, 按空格键继续",
                    color="white",
                    height=0.05,
                    wrapWidth=2,
                )
                start_msg.draw()
                self.win.flip()
                event.waitKeys(keyList=self.continue_keys)
                core.wait(0.3)
                self.win.flip()

                exp.entry(self.win, self.trialClock)

                end_msg = visual.TextStim(
                    self.win,
                    text="该实验结束, 按空格键继续",
                    color="white",
                    height=0.05,
                    wrapWidth=2,
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
            self.win.close()
        # core.quit()

    def pause(self):
        pause_msg = visual.TextStim(self.win, text="暂停中，按 r 恢复", height=0.20, wrapWidth=2)
        pause_start = self.trialClock.getTime()  # 记录暂停开始时间（系统时间）

        # 只用管理 win 和 clock 即可
        self.win.stashAutoDraw()
        paused = True
        while paused:
            pause_msg.draw()
            self.win.flip()
            keys = event.getKeys()
            if "r" in keys:  # 按 r 恢复
                paused = False

        # 恢复 win 的自动绘制
        self.win.retrieveAutoDraw()

        # === 校正时钟 ===
        self.trialClock.reset(-pause_start)

        # pause_end = core.getTime()
        # pause_duration = pause_end - pause_start
        # self.globalClock.addTime(-pause_duration)
        # self.trialClock.addTime(-pause_duration)


def run_session():
    # 切换到英文输入法
    switch_keyboard_layout()

    session = Session()
    exps = session.discover_experiments()
    selected = session.select_experiments_gui(exps)
    sort = session.sort_experiments(selected)
    session.add_experiments(sort)
    session.start(with_lsl=False)


if __name__ == "__main__":
    run_session()
