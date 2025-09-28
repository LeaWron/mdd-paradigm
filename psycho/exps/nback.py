import random

from psychopy import core, event, visual

from psycho.utils import init_lsl, send_marker

# ========== 参数设置 ==========
n_back = 2
n_blocks = 1
n_trials_per_block = 1
stim_pool = list(range(0, 4))
stim_duration = 1.0
resp_keys = ["space"]

# PsychoPy 全局对象
win = None
stim_text = None
clock = None

# 存储
stim_sequence = []
results = []

# LSL 全局对象
lsl_outlet = None

# ========== 生命周期函数 ==========


def pre_block(block_index: int):
    """block 开始前"""
    global stim_sequence
    stim_sequence = [random.choice(stim_pool) for _ in range(n_trials_per_block)]
    # 区块开始前的提示
    msg = visual.TextStim(
        win,
        text=f"准备进入第 {block_index + 1} 个区块\n按任意键开始",
        color="white",
        height=0.1,
        wrapWidth=2,
    )
    msg.draw()
    win.flip()
    event.waitKeys()

    send_marker(lsl_outlet, f"BLOCK_START_{block_index}", clock.getTime())


def block(block_index: int):
    """执行一个 block"""
    for trial_index in range(n_trials_per_block):
        pre_trial(trial_index)
        trial(trial_index)
        post_trial(trial_index)
    send_marker(lsl_outlet, f"BLOCK_END_{block_index}", clock.getTime())


def post_block(block_index):
    msg = visual.TextStim(
        win,
        text=f"第 {block_index + 1} 个区块结束\n休息一下\n按任意键继续",
        color="white",
        height=0.1,
        wrapWidth=2,
    )
    msg.draw()
    win.flip()
    event.waitKeys()


def pre_trial(t):
    """trial 开始前"""
    fixation = visual.TextStim(win, text="+", color="white", height=0.4, wrapWidth=2)
    fixation.draw()
    win.flip()
    core.wait(0.5)


def trial(t):
    """单个 trial 的逻辑"""
    global results

    stim = stim_sequence[t]
    stim_text.text = stim
    stim_text.draw()

    # 上横线
    line_top = visual.Line(win, start=(-0.15, 0.2), end=(0.15, 0.2), lineColor="white", lineWidth=10, units="norm")

    # 下横线
    line_bottom = visual.Line(win, start=(-0.15, -0.2), end=(0.15, -0.2), lineColor="white", lineWidth=10, units="norm")

    line_top.draw()
    line_bottom.draw()

    win.flip()
    stim_onset = clock.getTime()

    # 等待刺激期间响应
    keys = event.waitKeys(maxWait=stim_duration, keyList=resp_keys, timeStamped=True)

    # 判定 target
    is_target = False
    if t >= n_back and stim == stim_sequence[t - n_back]:
        is_target = True

    # 响应情况
    if keys:
        key, rt = keys[0]
        responded = True
        rt = rt - stim_onset
    else:
        responded = False
        rt = None

    # 正确与否
    if is_target and responded:
        correct = True
    elif (not is_target) and (not responded):
        correct = True
    else:
        correct = False

    if correct:
        line_top.color = "green"
        line_bottom.color = "green"
    else:
        line_top.color = "red"
        line_bottom.color = "red"

    line_top.draw()
    line_bottom.draw()
    stim_text.draw()
    win.flip()
    core.wait(0.5)

    if is_target:
        send_marker(lsl_outlet, f"TARGET_{rt}", stim_onset)
    else:
        send_marker(lsl_outlet, "NONTARGET", stim_onset)

    results.append([t, stim, is_target, responded, rt, correct])


def post_trial(t):
    """trial 结束后"""
    win.flip()
    # isi = get_isi()
    # core.wait(isi)
    core.wait(0.5)


def entry(win_session: visual.Window | None = None, clock_session: core.Clock | None = None):
    """实验入口"""
    global stim_text, lsl_outlet, win, clock
    win = win_session
    clock = clock_session

    # win = visual.Window(size=(800, 600), pos=(0, 0), fullscr=True, color="grey", units="pix")
    stim_text = visual.TextStim(win, text="", color="white", height=0.3, wrapWidth=2)

    lsl_outlet = init_lsl("NBackMarkers")  # 初始化 LSL

    for block_index in range(n_blocks):
        pre_block(block_index)
        block(block_index)
        post_block(block_index)

    # 实验结束

    send_marker(lsl_outlet, "EXPERIMENT_END", clock.getTime())


def main():
    entry()


if __name__ == "__main__":
    main()
