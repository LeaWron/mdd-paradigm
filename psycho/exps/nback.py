import random

from psychopy import core, event, visual

from psycho.utils import init_lsl, orbitary_keys, send_marker

# ========== 参数设置 ==========
n_back = 2
n_blocks = 1
n_trials_per_block = 10
stim_pool = list(range(0, 4))
stim_duration = 1.0
resp_keys = ["space"]
rest_duration = 30  # 每个 block 休息时间

# PsychoPy 全局对象
win = None
stim_text = None
clock = None

correct_count = 0  # 正确响应次数
# 存储
stim_sequence = []
results = []

# LSL 全局对象
lsl_outlet = None

# ========== 生命周期函数 ==========


def pre_block(block_index: int):
    """block 开始前"""
    global stim_sequence, correct_count
    stim_sequence = [random.choice(stim_pool) for _ in range(n_trials_per_block)]
    # 区块开始前的提示
    text = f"准备进入第 {block_index + 1} 个区块\n你有{rest_duration}秒响应时间\n或者可以按空格键直接开始"

    msg = visual.TextStim(
        win,
        text=text,
        color="white",
        height=0.1,
        wrapWidth=2,
    )
    msg.draw()
    win.flip()
    event.waitKeys(rest_duration, keyList=orbitary_keys)
    correct_count = 0

    send_marker(lsl_outlet, f"BLOCK_START_{block_index}", clock.getTime())


def block(block_index: int):
    """执行一个 block"""
    for trial_index in range(n_trials_per_block):
        pre_trial(trial_index)
        trial(trial_index)
        post_trial(trial_index)
    send_marker(lsl_outlet, f"BLOCK_END_{block_index}", clock.getTime())


def post_block(block_index: int):
    """block 结束后"""
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

    event.waitKeys(keyList=orbitary_keys)


def pre_trial(trial_index: int):
    """trial 开始前"""
    fixation = visual.TextStim(win, text="+", color="white", height=0.4, wrapWidth=2)
    fixation.draw()
    win.flip()
    core.wait(0.5)


def trial(t):
    """单个 trial 的逻辑"""
    global results, correct_count

    stim = stim_sequence[t]
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
        correct_count += 1
    elif (not is_target) and (not responded):
        correct = True
        correct_count += 1
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
    stim_left = stim_duration - (clock.getTime() - stim_onset)
    core.wait(stim_left)

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
    if win_session is None:
        win = visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")
    else:
        win = win_session

    if clock_session is None:
        clock = core.Clock()
    else:
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
