import random

from omegaconf import DictConfig
from psychopy import core, event, visual
from pylsl import StreamOutlet

from psycho.utils import arbitary_keys, init_lsl, send_marker

# === 参数设置 ===
nback = 2
n_blocks = 1
n_trials_per_block = 10
stim_pool = list(range(0, 4))

timing = {
    "fixation": 0.5,
    "stim": 1.0,
    "rest": 30,
}
resp_keys = ["space"]
continue_keys = ["space"]

# === 全局对象 ===
win = None
stim_text = None
clock = None
lsl_outlet = None

block_index = 0
trial_index = 0
correct_count = 0  # 正确响应次数
# 存储
stim_sequence = []
results = []


def pre_block():
    global stim_sequence, correct_count
    stim_sequence = [random.choice(stim_pool) for _ in range(n_trials_per_block)]
    # 区块开始前的提示
    text = f"准备进入第 {block_index + 1} 个区块," + (f"你有{timing['rest']}秒休息时间\n" if block_index > 0 else "\n") + "你可以按空格键直接开始"

    msg = visual.TextStim(
        win,
        text=text,
        color="white",
        height=0.1,
        wrapWidth=2,
    )
    msg.draw()
    win.flip()
    event.waitKeys(timing["rest"] if block_index > 0 else 5.0, keyList=continue_keys)
    correct_count = 0


def block():
    global trial_index
    for local_trial_index in range(n_trials_per_block):
        trial_index = local_trial_index
        pre_trial()
        trial()
        post_trial()


def post_block():
    # 区块结束后的提示
    correct_rate = correct_count / n_trials_per_block

    text = f"第 {block_index + 1} 个区块结束\n你共响应了 {correct_count} 次正确响应\n正确率为 {correct_rate * 100:.2f}%\n按任意键继续"

    msg = visual.TextStim(
        win,
        text=text,
        color="white",
        height=0.1,
        wrapWidth=2,
    )
    msg.draw()
    win.flip()

    event.waitKeys(5, keyList=arbitary_keys)


def pre_trial():
    fixation = visual.TextStim(win, text="+", color="white", height=0.4, wrapWidth=2)
    fixation.draw()
    win.flip()
    core.wait(timing["fixation"])


def trial():
    global results, correct_count

    stim = stim_sequence[trial_index]
    stim_text.text = stim
    stim_text.draw()

    # 上横线
    line_top = visual.Line(
        win,
        start=(-0.15, 0.2),
        end=(0.15, 0.2),
        lineColor="white",
        lineWidth=10,
        units="norm",
    )

    # 下横线
    line_bottom = visual.Line(
        win,
        start=(-0.15, -0.2),
        end=(0.15, -0.2),
        lineColor="white",
        lineWidth=10,
        units="norm",
    )

    line_top.draw()
    line_bottom.draw()

    send_marker(lsl_outlet, "TRIAL_START")
    win.flip()
    stim_onset = clock.getTime()

    # 等待刺激期间响应
    keys = event.waitKeys(maxWait=timing["stim"], keyList=resp_keys, timeStamped=True)

    # 判定 target
    is_target = False
    if trial_index >= nback and stim == stim_sequence[trial_index - nback]:
        is_target = True

    # 响应情况
    if keys:
        send_marker(lsl_outlet, "RESPONSE")
        key, rt = keys[0]
        responded = True
        rt = rt - stim_onset
    else:
        send_marker(lsl_outlet, "NO_RESPONSE")
        responded = False
        rt = None

    # 正确与否
    if is_target and responded:
        correct = True
        correct_count += 1
    elif (not is_target) and (not responded):
        correct = True
        correct_count += 1
    else:
        correct = False

    if correct and responded:
        line_top.color = "green"
        line_bottom.color = "green"
    elif not correct and responded:
        line_top.color = "red"
        line_bottom.color = "red"

    line_top.draw()
    line_bottom.draw()
    stim_text.draw()
    win.flip()
    stim_left = timing["stim"] - (clock.getTime() - stim_onset)
    core.wait(stim_left)

    if is_target:
        send_marker(lsl_outlet, "TARGET")
    else:
        send_marker(lsl_outlet, "NOT_TARGET")

    results.append([trial_index, stim, is_target, responded, rt, correct])


def post_trial():
    win.flip()
    core.wait(0.5)


def init_exp(config: DictConfig | None):
    def read_config(cfg: DictConfig):
        global nback, n_blocks, n_trials_per_block, timing, stim_pool
        if cfg is not None:
            n_blocks = cfg.n_blocks
            n_trials_per_block = cfg.n_trials_per_block
            timing = cfg.timing
            nback = cfg.nback
            stim_pool = list(range(cfg.n_stim))

    if config is not None:
        read_config(config)


def run_exp(cfg: DictConfig | None):
    """运行实验"""
    global block_index
    init_exp(cfg)

    if cfg is not None:
        prompt = visual.TextStim(
            win,
            text=cfg.phase_prompt,
            color="white",
            height=0.1,
            wrapWidth=2,
        )
        prompt.draw()
        win.flip()
        event.waitKeys(keyList=continue_keys)

    for local_block_index in range(n_blocks):
        block_index = local_block_index
        pre_block()
        block()
        post_block()


def entry(
    win_session: visual.Window | None = None,
    clock_session: core.Clock | None = None,
    lsl_outlet_session: StreamOutlet | None = None,
    config: DictConfig | None = None,
):
    """实验入口"""
    global stim_text, lsl_outlet, win, clock
    win = win_session if win_session else visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")

    clock = clock_session if clock_session else core.Clock()

    lsl_outlet = lsl_outlet_session if lsl_outlet_session else init_lsl("NBackMarker")  # 初始化 LSL
    stim_text = visual.TextStim(win, text="", color="white", height=0.3, wrapWidth=2)

    if config is not None and "pre" in config:
        run_exp(config.pre)

    send_marker(lsl_outlet, "EXPERIMENT_START")
    run_exp(config.full if config is not None else None)
    # 实验结束
    send_marker(lsl_outlet, "EXPERIMENT_END")


def main():
    entry()


if __name__ == "__main__":
    main()
