import random

from psychopy import core, event, visual
from pylsl import StreamInfo, StreamOutlet

from psycho.utils import init_lsl, send_marker

# ====== 参数设置 ======
n_blocks = 4  # block 数量
n_trials_per_block = 10  # 每个 block 的 trial 数
go_prob = 0.7  # Go trial 的概率
stim_duration = 1.0  # 刺激呈现时间
resp_keys = ["space"]  # 受试者按键
win = None  # 全局窗口对象
lsl_outlet = None


# 实验部分
def pre_block(block_index):
    msg = visual.TextStim(
        win, text=f"准备进入第 {block_index + 1} 个区块\n按任意键开始", color="white"
    )
    msg.draw()
    win.flip()
    event.waitKeys()

    send_marker(lsl_outlet, f"BLOCK_START_{block_index}")


def block(block_index: int):
    for trial_index in range(n_trials_per_block):
        pre_trial(trial_index)
        trial(trial_index)
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


def pre_trial(trial_index):
    # 空屏 + 注视点
    fixation = visual.TextStim(win, text="+", color="white")
    fixation.draw()
    win.flip()
    core.wait(0.5)


def trial(trial_index):
    # 随机决定 Go / No-Go
    is_go = random.random() < go_prob
    stim_text = "按键!" if is_go else "不要按!"
    stim = visual.TextStim(win, text=stim_text, color="white")
    stim.draw()
    win.flip()

    # trial 开始 marker
    send_marker(lsl_outlet, f"TRIAL_START_{trial_index}")
    send_marker(lsl_outlet, "STIM_GO" if is_go else "STIM_NOGO")
    # 反应
    keys = event.waitKeys(
        maxWait=stim_duration, keyList=resp_keys, timeStamped=core.Clock()
    )
    win.flip()
    # 反应 marker
    if keys:
        send_marker(lsl_outlet, f"RESPONSE_{keys[0][0]}_{keys[0][1]:.3f}")
    else:
        send_marker(lsl_outlet, "NO_RESPONSE")

    # 判断正确性
    if is_go and keys:  # Go 且按了键
        feedback = "正确!"
    elif is_go and not keys:  # Go 但没按
        feedback = "漏按!"
    elif not is_go and keys:  # No-Go 但按了
        feedback = "错误!"
    else:  # No-Go 且没按
        feedback = "正确!"

    # 显示反馈
    fb = visual.TextStim(win, text=feedback, color="yellow")
    fb.draw()
    win.flip()
    core.wait(0.5)
    # trial 结束 marker
    send_marker(lsl_outlet, f"TRIAL_END_{trial_index}")


def post_trial(trial_index):
    # 空屏间隔
    win.flip()
    # isi = get_isi()
    # core.wait(isi)
    core.wait(0.5)


def entry():
    global win, lsl_outlet
    win = visual.Window(
        size=(800, 600), pos=(0, 0), fullscr=True, color="grey", units="pix"
    )

    lsl_outlet = init_lsl("GoNogoMarkers")  # 初始化 LSL

    for block_index in range(n_blocks):
        pre_block(block_index)
        block(block_index)
        post_block(block_index)

    # 实验结束
    end_msg = visual.TextStim(win, text="该实验结束", color="white")
    end_msg.draw()
    win.flip()
    event.waitKeys()
    win.close()

    send_marker(lsl_outlet, "EXPERIMENT_END")


def main():
    entry()


# if __name__ == "__main__":
# main()
main()
