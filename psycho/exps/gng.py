import random

from psychopy import core, event, visual
from pylsl import StreamOutlet

from psycho.utils import arbitary_keys, get_isi, init_lsl, send_marker

# === 参数设置 ===
n_blocks = 1  # block 数量
n_trials_per_block = 10  # 每个 block 的 trial 数
go_prob = 0.7  # Go trial 的概率
resp_keys = ["space"]  # 受试者按键

fixation_duration = 0.5
total_trial_duration = 2.5  # 每个 trial 的总时间
rest_duration = 30  # 每个 block 休息时间

# === 全局变量 ===
block_index = 0
win = None  # 全局窗口对象
clock = None  # 全局时钟对象
lsl_outlet = None


# 实验部分
def pre_block():
    text = f"准备进入第 {block_index + 1} 个区块, 你有{rest_duration}秒休息时间\n或者可以按任意键开始"

    msg = visual.TextStim(
        win,
        color="white",
        text=text,
        height=0.1,
        wrapWidth=2,
        anchorHoriz="center",
        anchorVert="center",
    )
    msg.draw()
    win.flip()
    event.waitKeys(rest_duration, keyList=arbitary_keys)

    # send_marker(lsl_outlet, f"BLOCK_START_{block_index}")


def block():
    for trial_index in range(n_trials_per_block):
        pre_trial(trial_index)
        trial(trial_index)
        post_trial(trial_index)
    # send_marker(lsl_outlet, f"BLOCK_END_{block_index}")


def post_block():
    msg = visual.TextStim(
        win,
        text=f"第 {block_index + 1} 个区块结束\n按任意键继续",
        color="white",
        height=0.1,
        wrapWidth=2,
    )
    msg.draw()
    win.flip()
    event.waitKeys(1.0, keyList=arbitary_keys)


def pre_trial(trial_index):
    # 空屏 + 注视点
    fixation = visual.TextStim(win, text="+", color="white", height=0.4, wrapWidth=2)
    fixation.draw()
    win.flip()
    core.wait(fixation_duration)


def trial(trial_index):
    # 随机决定 Go / No-Go
    is_go = random.random() < go_prob

    win.flip()
    blank_duration = get_isi(0.5, 1.0)
    core.wait(blank_duration)

    # 可选: 结合图形, 颜色设置等
    # eclipse = visual.Circle(win, radius=0.5, edges=128, size=(0.8, 0.4), lineColor="black")
    # eclipse.draw()

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

    send_marker(lsl_outlet, "TRIAL_START")
    win.flip()

    # trial 开始 marker
    # send_marker(lsl_outlet, f"TRIAL_START_{trial_index}")
    # send_marker(lsl_outlet, "STIM_GO" if is_go else "STIM_NOGO")
    # 反应
    keys = event.waitKeys(
        maxWait=total_trial_duration - blank_duration,
        keyList=resp_keys,
        timeStamped=True,
    )
    # 反应 marker
    if keys:
        send_marker(lsl_outlet, "GO_RESPONSE")
    else:
        send_marker(lsl_outlet, "NOGO_NO_RESPONSE")

    # win.flip()
    # trial 结束 marker
    # send_marker(lsl_outlet, f"TRIAL_END_{trial_index}")


def post_trial(trial_index):
    # 空屏间隔
    win.flip()
    core.wait(0.5)


def entry(
    win_session: visual.Window | None = None,
    clock_session: core.Clock | None = None,
    lsl_outlet_session: StreamOutlet | None = None,
):
    global win, clock, lsl_outlet, block_index
    win = win_session if win_session else visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")

    clock = clock_session if clock_session else core.Clock()

    lsl_outlet = lsl_outlet_session if lsl_outlet_session else init_lsl("GoNogoMarker")  # 初始化 LSL

    send_marker(lsl_outlet, "EXPERIMENT_START")
    for local_block_index in range(n_blocks):
        block_index = local_block_index
        pre_block()
        block()
        post_block()

    # 实验结束
    send_marker(lsl_outlet, "EXPERIMENT_END")


def main():
    entry()


if __name__ == "__main__":
    main()
