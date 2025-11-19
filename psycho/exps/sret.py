import random
from pathlib import Path

from omegaconf import DictConfig
from psychopy import core, event, visual

from psycho.session import Experiment
from psycho.utils import (
    get_isi,
    init_lsl,
    save_csv_data,
    send_marker,
    setup_default_logger,
    update_block,
    update_trial,
)

# TODO: 中文词
# TODO: hydra 配置
# TODO: 生成序列
# 刺激以伪随机顺序呈现，连续呈现的相同效价的词语不超过两个。

# 在数据采集开始前，参与者使用情感中性的词语完成了三次练习试验

positive_words = ["美丽", "勇敢", "聪明", "有能力"]

negative_words = ["害怕", "孤独", "易怒", "痛苦"]

# TODO:需要在实际实验前用适当的形容词替换这个列表
distractor_words = [
    f"干扰{i}" for i in range(1, len(negative_words) + len(positive_words) + 1)
]

continue_keys = ["space"]
# === 全局配置 ===

timing = {
    "encoding": {
        "stim": 0.5,
        "fixation": 1.8,
        "question": 1.8,
        "response": 2.5,
        "iti": {
            "low": 1.5,
            "high": 1.7,
        },
    },
    "distractor": {
        "step": 1.0,  # 倒数计时的步长
    },
    "recall": {
        "duration": 20.0,  # 回忆任务给予的时间（秒）
    },
    "recognition": {
        "stim": 0.5,
        "max_resp": 3.0,  # 识别任务最大反应时间
    },
}

encoding_map = {"f": "yes", "j": "no"}
recognition_map = {"f": "old", "j": "new"}

phase_names = ["Encoding", "Distractor", "Recall", "Recognition"]

# === 全局变量 ===
win = None
clock = None
lsl_outlet = None
logger = None
pre = False
test = False

# 数据容器
data_to_save = {
    "exp_start_time": [],
    "exp_end_time": [],
    "phase": [],  # 区分 Encoding/Recall/Recognition
    "phase_start_time": [],
    "phase_end_time": [],
    "trial_start_time": [],
    "trial_end_time": [],
    "trial_index": [],
    "stim_word": [],
    "stim_type": [],  # Positive/Negative/Distractor
    "response": [],
    "rt": [],  # 仅计算认可的值
    "correct_answer": [],  # 用于识别任务 (old/new)
    "is_correct": [],  # 仅用于识别任务
}

one_trial_data = {key: None for key in data_to_save.keys()}
one_block_data = {key: [] for key in data_to_save.keys()}  # 这里的 block 指代 phase


def show_prompt(text, duration=None, wait_key=True):
    stim = visual.TextStim(
        win, text=text, color="white", wrapWidth=1.2, height=0.06, alignText="left"
    )
    stim.draw()
    win.flip()
    if wait_key:
        event.waitKeys(duration or float("inf"), keyList=continue_keys)
    elif duration:
        core.wait(duration)


def draw_fixation(time: float = None):
    fix = visual.TextStim(win, text="+", color="white", height=0.2)
    fix.draw()
    win.flip()
    core.wait(time or timing["fixation"])


def run_encoding_phase():
    phase_name = "Encoding"
    one_trial_data["phase"] = phase_name
    one_trial_data["phase_start_time"] = clock.getTime()

    show_prompt(
        f"任务一：自我描述判断\n\n屏幕将呈现一系列形容词。\n如果该词语符合对你自己的描述，请按{list(encoding_map.keys())[0]}键\n如果不符合，请按{list(encoding_map.keys())[1]}键。\n\n按空格键开始。"
    )

    # 准备刺激：40积极 + 40消极
    trials = []
    for w in positive_words:
        trials.append({"word": w, "type": "positive"})
    for w in negative_words:
        trials.append({"word": w, "type": "negative"})

    random.shuffle(trials)

    for idx, trial in enumerate(trials):
        one_trial_data["phase"] = phase_name
        one_trial_data["trial_index"] = idx
        one_trial_data["stim_word"] = trial["word"]
        one_trial_data["stim_type"] = trial["type"]
        one_trial_data["trial_start_time"] = clock.getTime()

        # 间隔
        draw_fixation(get_isi(timing["iti"]["low"], timing["iti"]["high"]))

        # Stimulus
        text_stim = visual.TextStim(win, text=trial["word"], color="white", height=0.15)
        text_stim.draw()
        win.flip()
        core.wait(timing["encoding_stim"])

        draw_fixation()

        prompt_stim = visual.TextStim(
            win,
            text="符合我(f)   不符合我(j)",
            pos=(0, 0),
            height=0.08,
            color="white",
        )

        prompt_stim.draw()
        win.flip()

        send_marker(lsl_outlet, "ENCODING_STIM_ONSET", is_pre=pre)
        onset_time = clock.getTime()

        keys = event.waitKeys(
            maxWait=timing["encoding_resp"],
            keyList=list(encoding_map.keys()),
            timeStamped=clock,
        )

        rt = 0
        resp = None

        if keys:
            key, timestamp = keys[0]
            rt = timestamp - onset_time
            resp = encoding_map[key]
            send_marker(lsl_outlet, f"RESPONSE_{resp.upper()}", is_pre=pre)

        one_trial_data["response"] = resp
        one_trial_data["rt"] = rt

        one_trial_data["trial_end_time"] = clock.getTime()
        # 保存该 trial 数据
        update_trial(one_trial_data, one_block_data)
        logger.info(f"Encoding Trial {idx}: {trial['word']} -> {resp}")

    update_block(one_block_data, data_to_save)


def run_distractor_phase():
    phase_name = "Distractor"
    show_prompt(
        "任务二：倒数任务\n\n屏幕上将出现数字。\n请跟随数字的节奏，在心中默数。\n\n按空格键开始。"
    )

    start_num = 50
    # 记录一次 Distractor 作为一个 trial 或者仅仅记录 log
    send_marker(lsl_outlet, "DISTRACTOR_START", is_pre=pre)

    for i in range(start_num, 0, -1):
        one_trial_data["phase"] = phase_name
        one_trial_data["stim_word"] = str(i)

        text_stim = visual.TextStim(win, text=str(i), color="yellow", height=0.3)
        text_stim.draw()
        win.flip()
        core.wait(timing["distractor_step"])  # 节奏控制

        # 简单的记录，虽不需要详细分析但保持格式一致
        update_trial(one_trial_data, one_block_data)

        # 允许按 ESC 退出
        if event.getKeys(keyList=["escape"]):
            break

    update_block(one_block_data, data_to_save)
    send_marker(lsl_outlet, "DISTRACTOR_END", is_pre=pre)


# TODO: 实现输入和记录
# 是否限制词库来辅助回忆


def run_recall_phase():
    """
    一个简单的 TextBox 收集所有输入。
    """
    phase_name = "Recall"
    intro_text = (
        "任务三：回忆任务\n\n"
        "请尽可能多地回忆刚才你在[任务一]见到的词。\n"
        "并通过键盘输入。\n"
        f"限时 {timing['recall_duration']} 秒。\n\n"
        "准备好后按空格键开始计时。"
    )
    show_prompt(intro_text)

    send_marker(lsl_outlet, "RECALL_START", is_pre=pre)

    timer = core.CountdownTimer(timing["recall_duration"])

    while timer.getTime() > 0:
        time_left = int(timer.getTime())
        msg = visual.TextStim(
            win, text=f"请回忆...\n\n剩余时间: {time_left} 秒", color="white"
        )
        msg.draw()
        win.flip()

        # ESC 跳过
        if event.getKeys(keyList=["escape"]):
            break
        core.wait(0.1)

    send_marker(lsl_outlet, "RECALL_END", is_pre=pre)

    # 记录一行数据表示完成
    one_trial_data["phase"] = phase_name
    one_trial_data["response"] = "Recall Phase Completed"
    update_trial(one_trial_data, one_block_data)
    update_block(one_block_data, data_to_save)

    show_prompt("回忆时间到！\n请停止回忆。\n按空格键进入下一阶段。")


def run_recognition_phase():
    phase_name = "Recognition"
    show_prompt(
        f"任务四：再认任务\n\n屏幕将出现一系列形容词。\n如果这个词刚才出现过，请按{list(recognition_map.keys())[0]}键。\n如果这个词是新出现的，请按{list(recognition_map.keys())[1]}键。\n\n按空格键开始。"
    )

    # 构造刺激：80个旧词 + 80个新词
    old_words = positive_words + negative_words
    new_words = distractor_words

    trials = []
    for w in old_words:
        trials.append(
            {
                "word": w,
                "type": "old_positive" if w in positive_words else "old_negative",
                "correct": "old",
            }
        )
    for w in new_words:
        trials.append({"word": w, "type": "distractor", "correct": "new"})

    random.shuffle(trials)

    for idx, trial in enumerate(trials):
        one_trial_data["phase"] = phase_name
        one_trial_data["trial_index"] = idx
        one_trial_data["stim_word"] = trial["word"]
        one_trial_data["stim_type"] = trial["type"]
        one_trial_data["correct_answer"] = trial["correct"]
        one_trial_data["trial_start_time"] = clock.getTime()

        draw_fixation(get_isi(timing["iti"]["low"], timing["iti"]["high"]))

        text_stim = visual.TextStim(win, text=trial["word"], color="white", height=0.15)
        text_stim.draw()
        win.flip()
        core.wait(timing["recognition_stim"])

        draw_fixation()

        prompt_stim = visual.TextStim(
            win, text="见过(f)   没见过(j)", pos=(0, 0), height=0.08, color="white"
        )

        prompt_stim.draw()
        win.flip()

        send_marker(lsl_outlet, "RECOG_STIM_ONSET", is_pre=pre)
        onset_time = clock.getTime()

        keys = event.waitKeys(
            keyList=list(recognition_map.keys()),
            maxWait=timing["recognition_max_resp"],
            timeStamped=clock,
        )

        rt = None
        resp_val = None
        is_correct = False

        if keys:
            key, timestamp = keys[0]
            rt = timestamp - onset_time
            resp_val = recognition_map[key]
            is_correct = resp_val == trial["correct"]
            send_marker(lsl_outlet, "RESPONSE", is_pre=pre)
        else:
            resp_val = "miss"
            send_marker(lsl_outlet, "MISS", is_pre=pre)

        one_trial_data["response"] = resp_val
        one_trial_data["rt"] = rt
        one_trial_data["is_correct"] = is_correct

        update_trial(one_trial_data, one_block_data)
        logger.info(
            f"Recog Trial {idx}: {trial['word']} (Truth:{trial['correct']}) -> Resp:{resp_val}, Correct:{is_correct}"
        )

    update_block(one_block_data, data_to_save)


def init_exp():
    pass


def run_exp():
    # 1. 编码任务
    run_encoding_phase()

    # 2. 干扰任务 (倒数)
    run_distractor_phase()

    # 3. 回忆任务
    run_recall_phase()

    # 4. 识别任务
    run_recognition_phase()


def entry(exp: Experiment | None = None):
    global win, clock, lsl_outlet, logger, pre, test

    win = exp.win or visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")
    clock = exp.clock or core.Clock()
    logger = exp.logger or setup_default_logger()
    lsl_outlet = exp.lsl_outlet or init_lsl("SRETMarker")
    test = exp.test

    # 是否需要预实验
    if exp.config is not None and "pre" in exp.config:
        pass

    logger.info("实验开始")
    one_trial_data["exp_start_time"] = clock.getTime()
    send_marker(lsl_outlet, "EXPERIMENT_START", is_pre=pre)

    run_exp()

    send_marker(lsl_outlet, "EXPERIMENT_END", is_pre=pre)
    one_trial_data["exp_end_time"] = clock.getTime()
    logger.info("实验结束")

    if exp.session_info:
        logger.info("保存数据")

        update_trial(one_trial_data, one_block_data)
        update_block(one_block_data, data_to_save)
        save_csv_data(data_to_save, exp.session_info["save_path"] + "-SRET")


def main():
    class MockExp:
        def __init__(self):
            self.win = None
            self.clock = None
            self.logger = None
            self.lsl_outlet = None
            self.test = True
            self.config = None

    entry(MockExp())


if __name__ == "__main__":
    main()
