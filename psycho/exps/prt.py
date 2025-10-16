import random

import numpy as np
from omegaconf import DictConfig
from psychopy import core, event, tools, visual
from pylsl import StreamOutlet

from psycho.utils import init_lsl, parse_stim_path, send_marker

# === 参数设置 ===
n_blocks = 1
n_trials_per_block = 10

timing = {
    "fixation": 0.5,
    "empty": 0.5,
    "stim": 0.1,
    "response": 0.5,
    "iti": 0.5,
    "feedback": 0.5,
    "rest": 30,
}
fixation_duration = 0.5

response_keys = ["s", "l"]
continue_keys = ["space"]

stim_folder = parse_stim_path("prt")
empty_face = stim_folder / "empty_face.png"
short_mouth = stim_folder / "short_mouth.png"
long_mouth = stim_folder / "long_mouth.png"

empty_duration = 0.5
stim_duration = 0.1

response_duration = 0.5

high_reward_prob = 0.8
low_reward_prob = 1 - high_reward_prob

feedback_duration = 0.5

iti = 0.5

rest_duration = 30

fov = 20  # 视场角, 单位: degree
monitor_distance = 60  # 显示器与人眼距离（单位：厘米）

reward_low = 1
reward_high = 10
reward_set = [reward_low, reward_high]
max_reward_count = 40  # 最大奖励次数
high_low_ratio = 3  # 高值奖励与低值奖励次数的比例
max_seq_same = 3  # 最大连续相同选择次数


# === 全局变量 ===
win = None
clock = None
lsl_outlet = None
block_index = 0
trial_index = 0

# 这里要随机选吗, 伪随机序列要生成两份吗
high_side = random.choice(response_keys)
total_point = 0
current_block_reward_count = 0

stim_sequences: dict[int, list] = None


# ========== 工具函数 ==========
# TODO: something about the reward
def give_reward(choice: str, high_side: str):
    """根据选择和当前高值侧决定奖励"""
    if choice == high_side:
        p = high_reward_prob
    else:
        p = low_reward_prob
    return reward_set[int(random.random() < p)]


# ========== 框架函数 ==========
def pre_block():
    global current_block_reward_count
    current_block_reward_count = max_reward_count

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
        text=f"第 {block_index + 1} 个区块结束\n你目前已有{total_point}分\n你有{rest_duration}秒休息时间\n你可以直接按空格键继续",
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
    core.wait(fixation_duration)


def trial():
    global total_point

    def show_stim():
        stim_size = get_stim_size()
        empty_face_stim = visual.ImageStim(
            win,
            image=empty_face,
            pos=(0, 0),
            size=(stim_size, stim_size),
            units="pix",
        )
        empty_face_stim.draw()
        win.flip()
        send_marker(lsl_outlet, "TRIAL_START")
        core.wait(empty_duration)

        long_or_short = random.choice(["long", "short"])
        if long_or_short == "short":
            short_mouth_stim = visual.ImageStim(
                win,
                image=short_mouth,
                pos=(0, 0),
                size=(stim_size, stim_size),
                units="pix",
            )
            short_mouth_stim.draw()
        else:
            long_mouth_stim = visual.ImageStim(
                win,
                image=long_mouth,
                pos=(0, 0),
                size=(stim_size, stim_size),
                units="pix",
            )
            long_mouth_stim.draw()
        win.flip()
        core.wait(stim_duration)
        return empty_face_stim, "s" if long_or_short == "short" else "l"

    empty_stim, long_or_short = show_stim()
    empty_stim.draw()
    win.flip()
    keys = event.waitKeys(maxWait=response_duration, keyList=response_keys, timeStamped=True)

    choice = "no_response"
    rt = None
    if keys:
        choice = keys[0][0]
        rt = keys[0][1]
        send_marker(lsl_outlet, "RESPONSE")
    else:
        send_marker(lsl_outlet, "NO_RESPONSE")

    if choice == long_or_short:
        reward = give_reward(choice, high_side)
    else:
        reward = 0

    # feedback
    if reward:
        feedback_reward = visual.TextStim(win, text=f"correct!\nYou won {reward} points", height=0.08, color="green" if reward == reward_high else "white")
        feedback_reward.draw()
        total_point += reward
    else:
        feedback_no = visual.TextStim(win, text="sorry, you got only 0 points", height=0.08, color="red")
        feedback_no.draw()
    win.flip()
    core.wait(feedback_duration)


def post_trial():
    # iti
    core.wait(iti)
    win.flip()


def get_stim_size() -> float:
    """根据显示器距离计算刺激大小"""

    # 计算刺激大小
    stim_size = 2 * np.tan(np.deg2rad(fov / 2)) * monitor_distance

    # stim_size = stim_size / (58.7 * 0.017455)
    stim_size = tools.monitorunittools.cm2pix(stim_size, win.monitor)  #  58.7 为 27 寸显示器宽度

    return stim_size  # 假设刺激大小与距离成比例


def init_exp(config: DictConfig | None):
    global \
        n_blocks, \
        n_trials_per_block, \
        timing, \
        stim_folder, \
        empty_face, \
        short_mouth, \
        long_mouth, \
        high_reward_prob, \
        monitor_distance, \
        fov, \
        reward_high, \
        reward_low, \
        reward_set, \
        max_reward_count, \
        high_low_ratio

    n_blocks = config.n_blocks
    n_trials_per_block = config.n_trials_per_block
    timing = config.timing
    stim_folder = parse_stim_path(config.stim_folder)
    empty_face = stim_folder / "empty_face.png"
    short_mouth = stim_folder / "short_mouth.png"
    long_mouth = stim_folder / "long_mouth.png"
    high_reward_prob = config.high_reward_prob
    monitor_distance = config.monitor_distance
    fov = config.fov
    reward_high = config.reward_high
    reward_low = config.reward_low
    reward_set = [reward_low, reward_high]
    max_reward_count = config.max_reward_count
    high_low_ratio = config.high_low_ratio


def run_exp(cfg: DictConfig | None):
    global block_index

    if cfg is not None:
        init_exp(cfg)
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
    global win, clock, lsl_outlet, block_index, port
    win = win_session if win_session else visual.Window(monitor="testMonitor", pos=(0, 0), fullscr=True, color="grey", units="norm")

    clock = clock_session if clock_session else core.Clock()

    lsl_outlet = lsl_outlet_session if lsl_outlet_session else init_lsl("PRTMarker")  # 初始化 LSL

    if config is not None and "pre" in config:
        run_exp(config.pre)

    send_marker(lsl_outlet, "EXPERIMENT_START")
    run_exp(config.full if config is not None else None)
    send_marker(lsl_outlet, "EXPERIMENT_END")


def main():
    entry()


if __name__ == "__main__":
    main()
