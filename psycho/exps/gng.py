import random

from psychopy import core, event, visual

from psycho.utils import init_lsl, orbitary_keys, send_marker

# ====== 参数设置 ======
n_blocks = 1  # block 数量
n_trials_per_block = 1  # 每个 block 的 trial 数
go_prob = 0.7  # Go trial 的概率
stim_duration = 1.0  # 刺激呈现时间
resp_keys = ["space"]  # 受试者按键

win = None  # 全局窗口对象
clock = None  # 全局时钟对象
lsl_outlet = None


# 实验部分
def pre_block(block_index):
    msg = visual.TextStim(
        win,
        text=f"准备进入第 {block_index + 1} 个区块\n按任意键开始",
        color="white",
        height=0.1,
        wrapWidth=2,
        anchorHoriz="center",
        anchorVert="center",
    )
    msg.draw()
    win.flip()
    event.waitKeys(keyList=orbitary_keys)

    send_marker(lsl_outlet, f"BLOCK_START_{block_index}", clock.getTime())


def block(block_index: int):
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
    event.waitKeys(keyList=orbitary_keys)


def pre_trial(trial_index):
    # 空屏 + 注视点
    fixation = visual.TextStim(win, text="+", color="white", height=0.4, wrapWidth=2)
    fixation.draw()
    win.flip()
    core.wait(0.5)


def trial(trial_index):
    # 随机决定 Go / No-Go
    is_go = random.random() < go_prob

    # TODO: 先显示一段时间的空白
    win.flip()
    core.wait(0.5)

    # TODO: 是否选择结合图形, 颜色设置等
    # ellipse = visual.Circle(win, radius=0.5, edges=128, size=(0.8, 0.4), lineColor="black")
    # ellipse.draw()

    stim_text = "按键!" if is_go else "不要按!"
    stim = visual.TextStim(
        win,
        text=stim_text,
        color="#f9f871" if is_go else "#ff6f91",
        height=0.3,
        wrapWidth=2,
        colorSpace="hex",
    )
    stim.draw()

    win.flip()

    # trial 开始 marker
    send_marker(lsl_outlet, f"TRIAL_START_{trial_index}", clock.getTime())
    send_marker(lsl_outlet, "STIM_GO" if is_go else "STIM_NOGO", clock.getTime())
    # 反应
    keys = event.waitKeys(maxWait=stim_duration, keyList=resp_keys, timeStamped=True)
    # 反应 marker
    if keys:
        send_marker(
            lsl_outlet, f"RESPONSE_{keys[0][0]}_{keys[0][1]:.3f}", clock.getTime()
        )
    else:
        send_marker(lsl_outlet, "NO_RESPONSE", clock.getTime())

    # 显示反馈
    # win.flip()
    core.wait(0.5)
    # trial 结束 marker
    send_marker(lsl_outlet, f"TRIAL_END_{trial_index}", clock.getTime())


def post_trial(trial_index):
    # 空屏间隔
    win.flip()
    # isi = get_isi()
    # core.wait(isi)
    core.wait(0.5)


def entry(
    win_session: visual.Window | None = None, clock_session: core.Clock | None = None
):
    global win, clock, lsl_outlet
    # win = visual.Window(size=(800, 600), pos=(0, 0), fullscr=True, color="grey", units="pix")
    win = win_session
    clock = clock_session

    lsl_outlet = init_lsl("GoNogoMarkers")  # 初始化 LSL

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
