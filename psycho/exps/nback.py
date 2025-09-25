import random

from psychopy import core, event, visual

from psycho.utils import check_exit, init_lsl, send_marker

# ========== 参数设置 ==========
n_back = 2
n_blocks = 4
n_trials_per_block = 30
stim_pool = list(range(0, 4))
stim_duration = 1.0
resp_keys = ["space"]

# PsychoPy 全局对象
win = None
stim_text = None

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
    msg = visual.TextStim(win, text=f"准备进入第 {block_index + 1} 个区块\n按任意键开始", color="white", height=30)
    msg.draw()
    win.flip()
    event.waitKeys()

    send_marker(lsl_outlet, f"BLOCK_START_{block_index}")


def block(block_index: int):
    """执行一个 block"""
    for trial_index in range(n_trials_per_block):
        check_exit()
        pre_trial(trial_index)
        check_exit()
        trial(trial_index)
        check_exit()
        post_trial(trial_index)
    send_marker(lsl_outlet, f"BLOCK_END_{block_index}")


def post_block(block_index):
    msg = visual.TextStim(
        win,
        text=f"第 {block_index + 1} 个区块结束\n休息一下\n按任意键继续",
        color="white",
    )
    msg.draw()
    win.flip()
    event.waitKeys()


def pre_trial(t):
    """trial 开始前"""
    fixation = visual.TextStim(win, text="+", color="white")
    fixation.draw()
    win.flip()
    core.wait(0.5)


def trial(t):
    """单个 trial 的逻辑"""
    global results

    stim = stim_sequence[t]
    stim_text.text = stim
    stim_text.draw()
    win.flip()
    stim_onset = core.getTime()

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

    results.append([t, stim, is_target, responded, rt, correct])


def post_trial(t):
    """trial 结束后"""
    win.flip()
    # isi = get_isi()
    # core.wait(isi)
    core.wait(0.5)


def entry():
    """实验入口"""
    global win, stim_text, lsl_outlet
    win = visual.Window(size=(800, 600), pos=(0, 0), fullscr=True, color="grey", units="pix")
    stim_text = visual.TextStim(win, text="wha", color="white", height=60)

    lsl_outlet = init_lsl("NBackMarkers")  # 初始化 LSL

    for block_index in range(n_blocks):
        check_exit()
        pre_block(block_index)
        check_exit()
        block(block_index)
        check_exit()
        post_block(block_index)

    # 实验结束
    end_msg = visual.TextStim(win, text="该实验结束", color="white")
    end_msg.draw()
    win.flip()
    event.waitKeys()
    win.close()

    send_marker("EXPERIMENT_END")
    core.quit()


def main():
    entry()


if __name__ == "__main__":
    main()
