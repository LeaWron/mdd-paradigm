import random

from psychopy import core, event, visual
from pylsl import StreamOutlet

from psycho.utils import init_lsl, send_marker

# === 参数设置 ===
n_blocks = 1
n_trials_per_block = 10

fixation_dur = 0.5

response_keys = ["left", "right"]
continue_keys = ["space"]

response_duration = 2.0
min_trials_before_reverse = 10
sliding_window = 10
stability_criterion = 8

high_reward_prob = 0.8
low_reward_prob = 1 - high_reward_prob

feedback_duration = 1.75

iti = 1.0

rest_duration = 30
# === 全局变量 ===
win = None
clock = None
lsl_outlet = None
block_index = 0
trial_index = 0

high_side = random.choice(response_keys)
choice_history = []
reverse_record = []


# ========== 工具函数 ==========
def give_reward(choice: str, high_side: str):
    """根据选择和当前高值侧决定奖励"""
    if choice == high_side:
        p = high_reward_prob
    else:
        p = low_reward_prob
    return random.random() < p


def check_reversal():
    """滑动窗口判据，决定是否反转"""
    global high_side, reverse_record
    if len(choice_history) < min(min_trials_before_reverse, sliding_window):
        return False
    last_window = choice_history[-sliding_window:]
    count_high = sum(1 for c in last_window if c == high_side)
    if count_high >= stability_criterion:
        high_side = "left" if high_side == "right" else "right"
        reverse_record.append((block_index, trial_index))
        return True
    return False


# ========== 框架函数 ==========
def pre_block():
    text = f"准备进入第 {block_index + 1} 个区块, 按空格键开始"
    msg = visual.TextStim(
        win,
        color="white",
        text=text,
        height=0.06,
        wrapWidth=2,
        anchorHoriz="center",
        anchorVert="center",
    )
    msg.draw()
    win.flip()
    event.waitKeys(5, keyList=continue_keys)

    send_marker(lsl_outlet, f"BLOCK_START_{block_index}")


def block():
    global trial_index
    for local_trial_index in range(n_trials_per_block):
        trial_index = local_trial_index
        pre_trial()
        trial()
        post_trial()


def post_block():
    msg = visual.TextStim(
        win,
        text=f"第 {block_index + 1} 个区块结束\n休息一下\n按空格键继续",
        color="white",
        height=0.06,
        wrapWidth=2,
    )
    msg.draw()
    win.flip()
    event.waitKeys(rest_duration, keyList=continue_keys)


def pre_trial():
    # fixation
    fixation = visual.TextStim(win, text="+", height=0.4, color="white")
    fixation.draw()
    win.flip()
    core.wait(fixation_dur)


def trial():
    def show_stim():
        left_stim = visual.Rect(
            win, width=0.2, height=0.2, pos=(-0.3, 0), color="steelblue"
        )
        left_stim.draw()
        right_stim = visual.Rect(
            win, width=0.2, height=0.2, pos=(0.3, 0), color="orange"
        )
        right_stim.draw()

    show_stim()
    win.flip()
    keys = event.waitKeys(
        maxWait=response_duration, keyList=response_keys, timeStamped=True
    )

    choice = "no_response"
    rt = None
    if keys:
        choice = keys[0][0]
        rt = keys[0][1]
        choice_history.append(choice)
    print(keys)

    reward = False
    if choice in ["left", "right"]:
        reward = give_reward(choice, high_side)

    # feedback
    feedback_reward = visual.TextStim(
        win, text="correct!\nYou won 10 points", height=0.08, color="yellow"
    )
    feedback_no = visual.TextStim(
        win, text="sorry, you got only 1 point", height=0.08, color="white"
    )
    if reward:
        feedback_reward.draw()
    else:
        feedback_no.draw()
    win.flip()
    core.wait(feedback_duration)

    # reversal check
    if choice in ["left", "right"]:
        check_reversal()
    send_marker(lsl_outlet, f"REWARD_{reward}_{rt}")


def post_trial():
    # iti
    core.wait(iti)
    win.flip()


def entry(
    win_session: visual.Window | None = None,
    clock_session: core.Clock | None = None,
    lsl_outlet_session: StreamOutlet | None = None,
):
    global win, clock, lsl_outlet, block_index, port
    win = (
        win_session
        if win_session
        else visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")
    )

    clock = clock_session if clock_session else core.Clock()

    lsl_outlet = (
        lsl_outlet_session if lsl_outlet_session else init_lsl("PRTMarker")
    )  # 初始化 LSL

    for local_block_index in range(n_blocks):
        block_index = local_block_index
        pre_block()
        block()
        post_block()

    send_marker(lsl_outlet, "EXPERIMENT_END", port=port)


def main():
    entry()


if __name__ == "__main__":
    main()
