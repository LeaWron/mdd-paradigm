import random
from pathlib import Path

from omegaconf import DictConfig
from psychopy import core, event, visual

from psycho.session import Experiment
from psycho.utils import (
    generate_trial_sequence,
    get_isi,
    init_lsl,
    save_csv_data,
    send_marker,
    setup_default_logger,
    switch_keyboard_layout,
    update_block,
    update_trial,
)

# === 参数设置 ===
positive_words = ["美丽", "勇敢", "聪明", "有能力"]

negative_words = ["害怕", "孤独", "易怒", "痛苦"]

distractor_words = [
    f"干扰{i}" for i in range(1, len(negative_words) + len(positive_words) + 1)
]

stim_sequence = {}

continue_keys = ["space"]

timing = {
    "encoding": {
        "stim": 0.5,
        "fixation": 1,
        "question": 1,
        "response": 2.5,
        "iti": {
            "low": 1,
            "high": 1.5,
        },
    },
    "distractor": {
        "start_num": 10,
        "step": 1.0,  # 倒数计时的步长
    },
    "recall": {
        "duration": 20.0,  # 回忆任务给予的时间（秒）
    },
    "recognition": {
        "stim": 0.5,
        "response": 3.0,  # 识别任务最大反应时间
        "iti": {
            "low": 1.2,
            "high": 1.5,
        },
    },
}

encoding_map = {"f": "yes", "j": "no"}
recognition_map = {"f": "old", "j": "new"}

phase_names = {
    "encoding": "Encoding",
    "distractor": "Distractor",
    "recall": "Recall",
    "recognition": "Recognition",
}

prompts = {
    "encoding": {
        "prompt": f"任务一：自我描述判断\n\n屏幕将呈现一系列形容词。\n如果该词语符合对你自己的描述，请按{list(encoding_map.keys())[0]}键\n如果不符合，请按{list(encoding_map.keys())[1]}键。\n\n按空格键开始。",
        "marker": "ENCODING",
    },
    "distractor": {
        "prompt": "任务二：倒数任务\n\n屏幕上将出现数字。\n请跟随数字的节奏，在心中默数。\n\n按空格键开始。",
        "marker": "DISTRACTOR",
    },
    "recall": {
        "prompt": f"任务三：回忆任务\n\n请尽可能多地回忆刚才你在[任务一]见到的词。\n并通过键盘输入。\n限时 {timing['recall']['duration']} 秒。\n\n准备好后按空格键开始计时。",
        "marker": "RECALL",
    },
    "recognition": {
        "prompt": f"任务四：再认任务\n\n屏幕将出现一系列形容词。\n如果这个词刚才出现过，请按{list(recognition_map.keys())[0]}键。\n如果这个词是新出现的，请按{list(recognition_map.keys())[1]}键。\n\n按空格键开始。",
        "marker": "RECOGNITION",
    },
}


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
    "stim_type": [],
    "response": [],
    "rt": [],
    "endorse_count": [],
    # 仅用于回忆任务
    "recalled_words": [],
    "recalled_all": [],
    "recall_count": [],
    # 用于识别任务
    "correct_answer": [],
    "is_correct": [],
    "recog_count": [],
}

one_trial_data = {key: None for key in data_to_save.keys()}
one_block_data = {key: [] for key in data_to_save.keys()}  # 这里的 block 指代 phase


# === 工具函数 ===
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


def draw_fixation(time: float):
    fix = visual.TextStim(win, text="+", color="white", height=0.2)
    fix.draw()
    win.flip()
    core.wait(time)


def init_encoding_phase():
    global stim_sequence
    if "encoding" not in stim_sequence:
        candidates = positive_words + negative_words
        stim_sequence["encoding"] = generate_trial_sequence(
            n_blocks=1,
            n_trials_per_block=len(candidates),
            stim_list=candidates,
        )[0]


def run_encoding_phase():
    send_marker(lsl_outlet, f"{prompts['encoding']['marker']}_START", is_pre=pre)
    one_trial_data["phase"] = phase_names["encoding"]
    one_trial_data["phase_start_time"] = clock.getTime()

    show_prompt(prompts["encoding"]["prompt"])

    trials = stim_sequence["encoding"]

    endorse_count = 0

    for idx, trial in enumerate(trials):
        one_trial_data["trial_index"] = idx
        one_trial_data["trial_start_time"] = clock.getTime()
        one_trial_data["stim_word"] = trial
        one_trial_data["stim_type"] = (
            "postive" if trial in positive_words else "negative"
        )

        # 间隔
        draw_fixation(
            get_isi(timing["encoding"]["iti"]["low"], timing["encoding"]["iti"]["high"])
        )

        # Stimulus
        text_stim = visual.TextStim(win, text=trial, color="white", height=0.15)
        text_stim.draw()
        win.flip()
        core.wait(timing["encoding"]["stim"])

        draw_fixation(timing["encoding"]["fixation"])

        prompt_stim = visual.TextStim(
            win,
            text="符合我(f)   不符合我(j)",
            pos=(0, 0),
            height=0.1,
            color="white",
        )

        prompt_stim.draw()
        win.flip()

        send_marker(lsl_outlet, "ENCODING_STIM_ONSET", is_pre=pre)
        onset_time = clock.getTime()

        keys = event.waitKeys(
            maxWait=timing["encoding"]["response"],
            keyList=list(encoding_map.keys()),
            timeStamped=clock,
        )

        rt = 0
        resp = None

        if keys:
            key, timestamp = keys[0]
            resp = encoding_map[key]
            if resp == "yes":
                rt = timestamp - onset_time
                endorse_count += 1
                one_trial_data["rt"] = rt
            send_marker(lsl_outlet, "RESPONSE", is_pre=pre)

        one_trial_data["response"] = resp

        one_trial_data["trial_end_time"] = clock.getTime()
        # 保存该 trial 数据
        update_trial(one_trial_data, one_block_data)
        logger.info(f"Encoding Trial {idx}: {trial} -> {resp}")

    one_trial_data["endorse_count"] = endorse_count
    one_trial_data["phase_end_time"] = clock.getTime()
    send_marker(lsl_outlet, f"{prompts['encoding']['marker']}_END", is_pre=pre)

    update_trial(one_trial_data, one_block_data)
    update_block(one_block_data, data_to_save)


def run_distractor_phase():
    send_marker(lsl_outlet, f"{prompts['distractor']['marker']}_START", is_pre=pre)
    show_prompt(prompts["distractor"]["prompt"])

    start_num = timing["distractor"]["start_num"]
    # 记录一次 Distractor 作为一个 trial 或者仅仅记录 log
    one_trial_data["phase"] = phase_names["distractor"]
    one_trial_data["phase_start_time"] = clock.getTime()

    for i in range(start_num, 0, -1):
        text_stim = visual.TextStim(win, text=str(i), color="yellow", height=0.3)
        text_stim.draw()
        win.flip()
        core.wait(timing["distractor"]["step"])  # 节奏控制

        # 允许按 ESC 退出
        if event.getKeys(keyList=["escape"]):
            break
    one_trial_data["phase_end_time"] = clock.getTime()
    send_marker(lsl_outlet, f"{prompts['distractor']['marker']}_END", is_pre=pre)

    update_trial(one_trial_data, one_block_data)
    update_block(one_block_data, data_to_save)


# [x]: 实现输入和记录
# 是否限制词库来辅助回忆
# 需要筛选词语, 保证拼音匹配
def run_recall_phase():
    send_marker(lsl_outlet, f"{prompts['recall']['marker']}_START", is_pre=pre)
    phase_name = phase_names["recall"]
    show_prompt(prompts["recall"]["prompt"])

    one_trial_data["phase"] = phase_name
    one_trial_data["phase_start_time"] = clock.getTime()
    # [ ] 测试是否需要取消全屏
    # win.fullscr = False

    # 存储所有提交的词
    submitted_words = []
    submitted_words_set = set()

    textbox = visual.TextBox2(
        win,
        text="",
        font="Microsoft YaHei",  # 用微软雅黑
        units="norm",
        pos=(0, 0),
        size=(1.0, 0.15),
        letterHeight=0.08,
        color="white",
        alignment="center",
        placeholder="请输入词汇并按回车提交",
        editable=True,
        borderColor="white",
        borderWidth=2,
        padding=0.02,
    )

    timer = core.CountdownTimer(timing["recall"]["duration"])

    # 界面元素
    tip_msg = visual.TextStim(
        win,
        text="请输入词汇并按回车提交",
        pos=(0, -0.3),
        height=0.05,
        font="Microsoft YaHei",
        color="white",
    )
    count_msg = visual.TextStim(
        win, text="已提交: 0", pos=(0, 0.3), height=0.05, font="Microsoft YaHei"
    )
    # 右上角显示剩余时间
    timer_msg = visual.TextStim(win, text="", pos=(0.7, 0.8), height=0.05)

    event.clearEvents()
    switch_keyboard_layout("zh-CN")

    # 主循环
    while timer.getTime() > 0:
        time_left = int(timer.getTime())
        timer_msg.text = f"剩余时间: {time_left}s"

        # 检查退出
        if event.getKeys(keyList=["escape"]):
            break

        # TextBox2 在 editable=True 时会捕获键盘输入。
        # 当按回车时，text 字段里会出现 '\n'。
        if "\n" in textbox.text:
            raw_text = textbox.text
            word = raw_text.strip()

            if word:  # 如果不是空行
                submitted_words.append(word)
                submitted_words_set.add(word)

                # [ ] 是否需要这里发送一个 marker 表示提交了一个词
                send_marker(lsl_outlet, "RECALL_SUBMIT", is_pre=pre)
                if word in submitted_words_set:
                    # [ ] 是否需要记录重复提交的词, 是否要给出提醒
                    logger.warning(f"重复提交了词: {word}")

            textbox.text = ""
            textbox.reset()

            # 更新计数显示
            count_msg.text = f"已提交: {len(submitted_words)}"

        tip_msg.draw()
        count_msg.draw()
        timer_msg.draw()
        textbox.draw()
        win.flip()

    send_marker(lsl_outlet, f"{prompts['recall']['marker']}_END", is_pre=pre)
    switch_keyboard_layout()

    # win.fullscr = True

    one_trial_data["recalled_words"] = ";".join(
        submitted_words_set & set(positive_words + negative_words)
    )
    one_trial_data["recalled_all"] = ";".join(submitted_words)
    one_trial_data["recall_count"] = len(submitted_words)
    one_trial_data["phase_end_time"] = clock.getTime()

    send_marker(lsl_outlet, f"{prompts['recall']['marker']}_END", is_pre=pre)
    logger.info(f"Recall Phase Completed. Words: {submitted_words}")

    update_trial(one_trial_data, one_block_data)
    update_block(one_block_data, data_to_save)

    show_prompt(
        f"回忆阶段结束！\n你一共提交了 {len(submitted_words)} 个词。\n按空格键进入下一阶段。"
    )


def init_recognition_phase():
    global stim_sequence
    if "recognition" not in stim_sequence:
        stim_list = positive_words + negative_words + distractor_words
        stim_sequence["recognition"] = generate_trial_sequence(
            n_blocks=1,
            n_trials_per_block=len(stim_list),
            stim_list=stim_list,
        )[0]


def run_recognition_phase():
    send_marker(lsl_outlet, f"{prompts['recognition']['marker']}_START", is_pre=pre)

    phase_name = phase_names["recognition"]
    one_trial_data["phase"] = phase_name
    one_trial_data["phase_start_time"] = clock.getTime()
    show_prompt(prompts["recognition"]["prompt"])

    trials = stim_sequence["recognition"]

    old_stim_list = positive_words + negative_words
    recog_count = 0

    for idx, trial in enumerate(trials):
        one_trial_data["trial_index"] = idx
        one_trial_data["stim_word"] = trial
        one_trial_data["stim_type"] = "old" if trial in old_stim_list else "new"
        one_trial_data["trial_start_time"] = clock.getTime()

        draw_fixation(
            get_isi(
                timing["recognition"]["iti"]["low"],
                timing["recognition"]["iti"]["high"],
            )
        )

        text_stim = visual.TextStim(win, text=trial, color="white", height=0.15)
        text_stim.draw()
        win.flip()
        core.wait(timing["recognition"]["stim"])

        draw_fixation(
            get_isi(
                timing["recognition"]["iti"]["low"],
                timing["recognition"]["iti"]["high"],
            )
        )

        prompt_stim = visual.TextStim(
            win, text="见过(f)   没见过(j)", pos=(0, 0), height=0.1, color="white"
        )

        prompt_stim.draw()
        win.flip()

        send_marker(lsl_outlet, "RECOG_STIM_ONSET", is_pre=pre)
        onset_time = clock.getTime()

        keys = event.waitKeys(
            keyList=list(recognition_map.keys()),
            maxWait=timing["recognition"]["response"],
            timeStamped=clock,
        )

        rt = None
        resp_val = None
        is_correct = False

        if keys:
            key, timestamp = keys[0]
            rt = timestamp - onset_time
            one_trial_data["rt"] = rt
            resp_val = recognition_map[key]
            is_correct = resp_val == one_trial_data["stim_type"]
            send_marker(lsl_outlet, "RESPONSE", is_pre=pre)
        else:
            resp_val = "noresp"
            send_marker(lsl_outlet, "NORESPONSE", is_pre=pre)

        one_trial_data["response"] = resp_val
        one_trial_data["is_correct"] = is_correct

        if is_correct and resp_val == "old":
            one_trial_data["correct_answer"] = trial
            recog_count += 1

        one_trial_data["trial_end_time"] = clock.getTime()
        logger.info(
            f"Recog Trial {idx}: {trial} (Truth:{one_trial_data['stim_type']}) -> Resp:{resp_val}, Correct:{is_correct}"
        )
        update_trial(one_trial_data, one_block_data)

    one_trial_data["recog_count"] = recog_count
    one_trial_data["phase_end_time"] = clock.getTime()
    send_marker(lsl_outlet, f"{prompts['recognition']['marker']}_END", is_pre=pre)

    update_trial(one_trial_data, one_block_data)
    update_block(one_block_data, data_to_save)


def init_exp(config: DictConfig | None):
    global timing, phase_names, prompts, stim_sequence, data_to_save

    phase_names = config["phase_names"]
    prompts = config["prompts"]

    if pre or test is False:
        timing = config["timing"]
    if "stim_sequence" in config:
        stim_sequence = config.stim_sequence
    if "stims" in config:
        global positive_words, negative_words, distractor_words

        if not pre:
            positive_words = config.stims["positive"][:10]
            negative_words = config.stims["negative"][:10]
            distractor_words = config.stims["distractor"][:len(positive_words)+len(negative_words)]
        else:
            total_stims = config.stims["neutral"]
            len_total_stims = len(total_stims)
            positive_words = total_stims[: len_total_stims // 4]
            negative_words = total_stims[len_total_stims // 4 : len_total_stims // 2]
            distractor_words = total_stims[len_total_stims // 2 :]

    for key in data_to_save.keys():
        data_to_save[key].clear()


def run_exp(cfg: DictConfig | None):
    if cfg is not None:
        init_exp(cfg)
        prompt = visual.TextStim(
            win,
            text=cfg.phase_prompt,
            color="white",
            height=0.06,
            wrapWidth=1.2,
            pos=(0, 0),
            alignText="left",
        )
        prompt.draw()
        win.flip()
        event.waitKeys(keyList=continue_keys)

    init_encoding_phase()
    init_recognition_phase()
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
        pre = True
        while True:
            run_exp(exp.config.pre)

            commit_text = (
                "是否需要再次进行预实验?\n按 y 键再次进行预实验, 按 n 键结束预实验"
            )
            prompt = visual.TextStim(
                win,
                text=commit_text,
                color="white",
                height=0.1,
                wrapWidth=2,
            )
            prompt.draw()
            win.flip()
            keys = event.waitKeys(keyList=["y", "n"])
            if keys and keys[0] == "n":
                break
        pre = False

    logger.info("实验开始")
    one_trial_data["exp_start_time"] = clock.getTime()
    send_marker(lsl_outlet, "EXPERIMENT_START", is_pre=pre)

    run_exp(exp.config.full if exp.config else None)

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
            self.session_info = None

    switch_keyboard_layout()
    entry(MockExp())


if __name__ == "__main__":
    main()
