import random

import numpy as np
from omegaconf import DictConfig
from psychopy import core, event, tools, visual

from psycho.session import PSYCHO_FONT, Experiment
from psycho.utils import (
    adapt_image_stim_size,
    init_lsl,
    parse_stim_path,
    save_csv_data,
    send_marker,
    setup_default_logger,
    update_block,
    update_trial,
)

# === 参数设置 ===
n_blocks = 3
n_trials_per_block = 20

timing = {
    "fixation": 0.5,
    "empty": 0.5,
    "stim": 0.1,
    "response": 0.5,
    "iti": 0.5,
    "feedback": 0.5,
    "rest": 5,
    "show": 5,
}

response_keys = ["s", "l"]
continue_keys = ["space"]

stim_folder = parse_stim_path("prt")
empty_face = stim_folder / "empty_face.png"
short_mouth = stim_folder / "short_mouth.png"
long_mouth = stim_folder / "long_mouth.png"


high_reward_prob = 0.6
low_reward_prob = 1 - high_reward_prob


fov = 8  # 视场角, 单位: degree
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
logger = None

block_index = 0
trial_index = 0

high_side_abrr = random.choice(response_keys)
if high_side_abrr == "s":
    high_side = "short"
    low_side = "long"
else:
    high_side = "long"
    low_side = "short"

total_point = 0
correct_count = 0

stim_sequence: dict[int, list] = None
reward_indice = None
high_cache = 0
low_cache = 0

pre = 0
test = False

# === 数据保存 ===
data_to_save = {
    "exp_start_time": [],
    "exp_end_time": [],
    "block_index": [],
    "trial_index": [],
    "trial_start_time": [],
    "trial_end_time": [],
    "stim": [],
    "choice": [],
    "rt": [],
    "correct_rate": [],
    "reward": [],
    "total_point": [],
}

one_trial_data = {key: None for key in data_to_save.keys()}
one_block_data = {key: [] for key in data_to_save.keys()}


# ========== 工具函数 ==========
def give_reward(choice: str, right_choice: str):
    """根据选择和当前高值侧决定奖励"""
    if reward_indice is not None:
        global high_cache, low_cache
        if choice == right_choice:
            if trial_index in reward_indice[block_index]["high"]:
                return reward_high
            elif trial_index in reward_indice[block_index]["low"]:
                return reward_low
            elif right_choice == high_side_abrr and high_cache > 0:
                high_cache -= 1
                return reward_high
            elif right_choice != high_side_abrr and low_cache > 0:
                low_cache -= 1
                return reward_low
            else:
                return 0
        else:
            if trial_index in reward_indice[block_index]["high"]:
                high_cache += 1
            elif trial_index in reward_indice[block_index]["low"]:
                low_cache += 1
            return -1
    else:
        if choice == right_choice:
            if choice == high_side_abrr and random.random() < high_reward_prob:
                return reward_high
            elif choice != high_side_abrr and random.random() < low_reward_prob:
                return reward_low
            else:
                return 0
        else:
            return -1


# ========== 框架函数 ==========
def pre_block():
    text_front = f"当前区块为第 {block_index + 1} 个区块\n" if not pre else ""
    text = f"{text_front}按<c=#51d237>空格键</c>开始"
    msg = visual.TextBox2(
        win,
        color="white",
        text=text,
        letterHeight=0.08,
        size=(1.2, None),
        font=PSYCHO_FONT,
        alignment="center",
    )
    msg.draw()
    win.flip()
    event.waitKeys(5, keyList=continue_keys)


def block():
    global trial_index
    for local_trial_index in range(n_trials_per_block):
        trial_index = local_trial_index
        one_trial_data["trial_index"] = trial_index

        # 记录实验开始时间
        trial_start_time = clock.getTime()
        one_trial_data["trial_start_time"] = trial_start_time
        if pre > 1:
            key = event.getKeys(["escape"])
            if key:
                return
        pre_trial()
        if pre > 1:
            key = event.getKeys(["escape"])
            if key:
                return
        trial()
        post_trial()

        # 记录实验结束时间
        trial_end_time = clock.getTime()
        one_trial_data["trial_end_time"] = trial_end_time

        update_trial(one_trial_data, one_block_data)


def post_block():
    global correct_count
    logger.info(f"Block {block_index + 1} end, total point: {total_point}")
    one_trial_data["total_point"] = total_point
    one_trial_data["correct_rate"] = 1.0 * correct_count / n_trials_per_block
    correct_count = 0

    logger.info(
        f"Block {block_index + 1} end, current block correct rate: {one_trial_data['correct_rate'] * 100:.2f}%, total point: {total_point}"
    )
    update_trial(one_trial_data, one_block_data)

    text_front = "" if pre else "该区块结束\n"
    rest_time = timing["rest"]
    for i in range(rest_time, -1, -1):
        msg = visual.TextBox2(
            win,
            color="white",
            text=f"{text_front}你目前已有 <c=yellow>{total_point}</c> 分\n你有 <c=yellow>{i}</c> 秒休息时间\n按<c=#51d237>空格键</c>继续",
            letterHeight=0.08,
            size=(1.2, None),
            font=PSYCHO_FONT,
            alignment="center",
        )
        msg.draw()
        win.flip()
        if event.waitKeys(1, keyList=continue_keys):
            break


def pre_trial():
    # fixation
    fixation = visual.TextStim(
        win, text="+", height=0.2, color="white", font=PSYCHO_FONT
    )
    fixation.draw()
    win.flip()
    core.wait(timing["fixation"])


def trial():
    global total_point, correct_count

    def show_stim():
        empty_face_stim = visual.ImageStim(
            win,
            image=empty_face,
            pos=(0, 0),
            size=fov,
            units="deg",
        )
        empty_face_stim.draw()
        win.flip()
        send_marker(lsl_outlet, "TRIAL_START", is_pre=pre)
        core.wait(timing["empty"])

        if stim_sequence is not None:
            high_or_low = stim_sequence[block_index][trial_index]
            if high_or_low == "high":
                cur_side = high_side
            else:
                cur_side = low_side
        else:
            cur_side = random.choice(["long", "short"])
        if cur_side == "long":
            cur_stim = long_mouth
        else:
            cur_stim = short_mouth
        mouth_stim = visual.ImageStim(
            win,
            image=cur_stim,
            pos=(0, 0),
            size=fov,
            units="deg",
        )
        mouth_stim.draw()
        win.flip()
        on_set = clock.getTime()
        core.wait(timing["stim"])
        return empty_face_stim, on_set, "s" if cur_side == "short" else "l"

    empty_stim, on_set, long_or_short = show_stim()
    empty_stim.draw()
    win.flip()

    one_trial_data["stim"] = long_or_short
    keys = event.waitKeys(
        maxWait=timing["response"], keyList=response_keys, timeStamped=clock
    )

    choice = "no_response"
    rt = None
    if keys:
        choice = keys[0][0]
        rt = keys[0][1] - on_set
        # 记录反应时
        one_trial_data["choice"] = choice
        one_trial_data["rt"] = rt

        send_marker(lsl_outlet, "RESPONSE", is_pre=pre)
        logger.info(
            f"Block {block_index + 1}, trial {trial_index + 1}: Correct face: {long_or_short}, Response: {choice}, rt: {rt:.4f}"
        )
    else:
        send_marker(lsl_outlet, "NO_RESPONSE", is_pre=pre)
        logger.info("No response")

    reward = give_reward(choice, long_or_short)
    # 记录奖励
    one_trial_data["reward"] = reward
    logger.info(f"Reward: {reward}")

    # feedback
    if reward > 0:
        feedback_reward = visual.TextStim(
            win,
            text=f"你获得了 {reward} 分!",
            height=0.1,
            color="#51d237",
            colorSpace="rgb",
            font=PSYCHO_FONT,
        )
        feedback_reward.draw()
        total_point += reward
        correct_count += 1
        # 显示总分数
        visual.TextBox2(
            win,
            text=f"你当前的分数为 <c=yellow>{total_point}</c>",
            letterHeight=0.08,
            size=(1.2, None),
            font=PSYCHO_FONT,
            pos=(0, -0.2),
            color="white",
            alignment="center",
        ).draw()
    elif reward == 0:
        # feedback_no = visual.TextStim(
        #     win,
        #     text="正确!",
        #     height=0.1,
        #     color="white",
        #     font=PSYCHO_FONT,
        # )
        # feedback_no.draw()
        correct_count += 1
    # elif reward < 0:
    #     feedback_wrong = visual.TextStim(
    #         win,
    #         text="错误!" if rt is not None else "超时!",
    #         height=0.1,
    #         color="#eb5555",
    #         colorSpace="rgb",
    #         font=PSYCHO_FONT,
    #     )
    #     feedback_wrong.draw()
    if pre and rt is None:
        feedback_wrong = visual.TextStim(
            win,
            text="超时!请尽快作出反应!",
            height=0.1,
            color="#eb5555",
            colorSpace="rgb",
            font=PSYCHO_FONT,
        )
        feedback_wrong.draw()
    #

    win.flip()
    core.wait(timing["feedback"])


def post_trial():
    # iti
    win.flip()
    core.wait(timing["iti"])


def get_stim_size() -> float:
    """根据显示器距离计算刺激大小"""

    # 计算刺激大小
    stim_size = 2 * np.tan(np.deg2rad(fov / 2)) * monitor_distance

    # stim_size = stim_size / (58.7 * 0.017455)
    stim_size = tools.monitorunittools.cm2pix(
        stim_size, win.monitor
    )  #  58.7 为 27 寸显示器宽度

    return stim_size  # 假设刺激大小与距离成比例


def show_stims():
    prompt = visual.TextBox2(
        win,
        text="现在为你展示会出现的脸的图片\n从左到右依次为: 空脸, 短嘴巴, 长嘴巴",
        letterHeight=0.08,
        size=(2, None),
        pos=(0, 0.6),
        color="white",
        font=PSYCHO_FONT,
        alignment="center",
    )
    prompt.draw()

    stim_height, aspect_ratio = adapt_image_stim_size(win, empty_face, 0.3)
    stim_width = stim_height * aspect_ratio
    empty_face_stim = visual.ImageStim(
        win,
        image=empty_face,
        pos=(-0.5, 0),
        size=(stim_width, stim_height),
        units="norm",
    )

    short_mouth_stim = visual.ImageStim(
        win,
        image=short_mouth,
        pos=(0, 0),
        size=(stim_width, stim_height),
        units="norm",
    )
    long_mouth_stim = visual.ImageStim(
        win,
        image=long_mouth,
        pos=(0.5, 0),
        size=(stim_width, stim_height),
        units="norm",
    )

    empty_face_stim.draw()
    short_mouth_stim.draw()
    long_mouth_stim.draw()

    visual.TextBox2(
        win,
        text="观察完毕后按<c=#51d237>空格键</c>继续",
        letterHeight=0.08,
        size=(2, None),
        pos=(0, -0.6),
        color="white",
        font=PSYCHO_FONT,
        alignment="center",
    ).draw()

    win.flip()
    event.waitKeys(keyList=continue_keys)


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
        high_low_ratio, \
        stim_sequence, \
        reward_indice, \
        data_to_save, \
        total_point, \
        correct_count

    if pre or not test:
        logger.info("Run in real exp")

        n_blocks = config.n_blocks
        n_trials_per_block = config.n_trials_per_block
        timing = config.timing

    stim_folder = parse_stim_path(config.stim_folder)
    empty_face = stim_folder / "empty_face.png"
    short_mouth = stim_folder / "short_mouth.png"
    long_mouth = stim_folder / "long_mouth.png"
    monitor_distance = config.monitor_distance
    fov = config.fov
    reward_high = config.reward_high
    reward_low = config.reward_low
    reward_set = [reward_low, reward_high]
    max_reward_count = config.max_reward_count
    high_low_ratio = config.high_low_ratio

    if "stim_sequence" in config:
        stim_sequence = config.stim_sequence

    if "reward_indice" in config:
        reward_indice = config.reward_indice

    for key in data_to_save.keys():
        data_to_save[key].clear()
    total_point = 0
    correct_count = 0


def run_exp(cfg: DictConfig | None):
    global block_index

    if cfg is not None:
        init_exp(cfg)
        title = visual.TextBox2(
            win,
            text=cfg.phase_prompt.title,
            color="white",
            letterHeight=0.08,
            size=(2, None),
            pos=(0, 0.7),
            alignment="center",
            font=PSYCHO_FONT,
        )
        prompt = visual.TextBox2(
            win,
            text=cfg.phase_prompt.prompt,
            color="white",
            letterHeight=cfg.phase_prompt.letterHeight or 0.06,
            size=(
                cfg.phase_prompt.size.width or 1.5,
                cfg.phase_prompt.size.height or 1.5,
            ),
            pos=(0, 0),
            alignment=cfg.phase_prompt.alignment or "left",
            font=PSYCHO_FONT,
        )
        title.draw()
        prompt.draw()
        win.flip()
        event.waitKeys(keyList=continue_keys)

        if pre:
            show_stims()

    for local_block_index in range(n_blocks):
        block_index = local_block_index
        one_trial_data["block_index"] = block_index

        pre_block()
        block()
        post_block()

        update_block(one_block_data, data_to_save)


def entry(exp: Experiment | None = None):
    global win, clock, lsl_outlet, block_index, logger, pre, test
    win = exp.win or visual.Window(
        monitor="testMonitor", pos=(0, 0), fullscr=True, color="grey", units="norm"
    )

    clock = exp.clock or core.Clock()
    logger = exp.logger or setup_default_logger()

    lsl_outlet = exp.lsl_outlet or init_lsl("PRTMarker")  # 初始化 LSL

    test = exp.test

    if exp.config is not None and "pre" in exp.config:
        pre = 1
        while True:
            run_exp(exp.config.pre)

            commit_text = "预实验已完成\n你是否需要再次进行预实验以更熟悉任务?\n按 <c=#51d237>Y</c> 键再次进行预实验, 按 <c=#eb5555>N</c> 键进入正式实验\n"
            prompt = visual.TextBox2(
                win,
                text=commit_text,
                color="white",
                letterHeight=0.08,
                size=(2, None),
                alignment="center",
                pos=(0, 0),
                font=PSYCHO_FONT,
            )

            prompt.draw()
            win.flip()
            keys = event.waitKeys(keyList=["y", "n"])
            if keys and keys[0] == "n":
                break
            pre += 1
            tip = visual.TextBox2(
                win,
                text="如果你认为已经充分熟悉实验,可以在每个试次前按 <c=#51d237>ESC</c> 键退出预实验\n\n按<c=#51d237>空格键</c>进入预实验",
                color="white",
                letterHeight=0.08,
                size=(2, None),
                alignment="center",
                pos=(0, 0),
                font=PSYCHO_FONT,
            )
            tip.draw()
            win.flip()
            event.waitKeys(keyList=continue_keys)
        pre = 0

    # 记录实验开始时间
    one_trial_data["exp_start_time"] = clock.getTime()

    send_marker(lsl_outlet, "EXPERIMENT_START", is_pre=pre)
    logger.info("实验开始")

    run_exp(exp.config.full if exp.config is not None else None)

    send_marker(lsl_outlet, "EXPERIMENT_END", is_pre=pre)
    logger.info("实验结束")
    # 记录实验结束时间
    one_trial_data["exp_end_time"] = clock.getTime()

    if exp.config is not None:
        logger.info("保存数据")
        update_trial(one_trial_data, one_block_data)
        update_block(one_block_data, data_to_save)

        save_csv_data(
            data_to_save,
            exp.session_info["save_path"] + "-prt",
            exp.session_info["group"],
        )


def main():
    entry(Experiment(None, None, None, None, None, None))


if __name__ == "__main__":
    main()
