from typing import Literal

from omegaconf import DictConfig
from psychopy import core, event, visual

from psycho.session import PSYCHO_FONT, Experiment
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

encoding_map = {"a": "yes", "d": "no"}
recognition_map = {"f": "old", "j": "new"}

phase_names = {
    "encoding": "Encoding",
    "distractor": "Distractor",
    "recall": "Recall",
    "recognition": "Recognition",
}

intensity_prompt = "请评估该词汇对您个人特征的描述程度"
intensity_ticks = list(range(1, 9 + 1))


# === 全局变量 ===
win: visual.Window = None
clock: core.Clock = None
lsl_outlet = None
logger = None
pre = 0
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
    "intensity": [],
    "endorse_count": [],
    # # 仅用于回忆任务
    # "recalled_words": [],
    # "recalled_all": [],
    # "recall_count": [],
    # # 用于识别任务
    # "correct_answer": [],
    # "is_correct": [],
    # "recog_count": [],
}

one_trial_data = {key: None for key in data_to_save.keys()}
one_block_data = {key: [] for key in data_to_save.keys()}  # 这里的 block 指代 phase


# === 工具函数 ===
def show_prompt(text, duration=None, wait_key=True):
    stim = visual.TextBox2(
        win,
        text=text,
        color="white",
        letterHeight=0.08,
        size=(1.2, None),
        font=PSYCHO_FONT,
        alignment="left",
    )
    stim.draw()
    win.flip()
    if wait_key:
        event.waitKeys(duration or float("inf"), keyList=continue_keys)
    elif duration:
        core.wait(duration)


def draw_fixation(time: float):
    fix = visual.TextStim(win, text="+", color="white", height=0.2, font=PSYCHO_FONT)
    fix.draw()
    win.flip()
    core.wait(time)


def rating_slider(resp: Literal["yes", "no"]):
    win.setMouseVisible(visibility=True)
    prompt = visual.TextBox2(
        win,
        text=intensity_prompt,
        color="white",
        pos=(0, 0.4),
        size=(1.2, None),
        letterHeight=0.06,
        alignment="center",
        font=PSYCHO_FONT,
    )
    slider = visual.Slider(
        win,
        ticks=intensity_ticks,
        labels=["0%", "50%", "100%"],
        granularity=0,  # 连续可拖动
        # startValue=labels[len(labels) // 2],
        size=(0.9, 0.05),
        style=["rating"],
        color="white",
        pos=(0, 0),
        font=PSYCHO_FONT,
        labelHeight=0.04,
    )
    # slider.setValue(slider.startValue)

    # 按钮
    base_color = [-0.2, -0.2, -0.2]  # 比灰背景稍深
    hover_color = [0.1, 0.1, 0.1]  # 鼠标悬停时略亮
    disabled_color = [-0.5, -0.5, -0.5]

    button_box = visual.Rect(
        win,
        width=0.4,
        height=0.3,
        pos=(0, -0.5),
        fillColor=disabled_color,
        lineColor="white",
    )
    button_text = visual.TextStim(
        win, text="确认", color="white", pos=(0, -0.5), height=0.08
    )

    mouse = event.Mouse(win=win)
    # 选择中性的时候不打分
    while True:
        prompt.draw()
        slider.draw()

        if slider.markerPos is None:
            button_box.fillColor = disabled_color
        else:
            if button_box.contains(mouse):
                button_box.fillColor = hover_color
            else:
                button_box.fillColor = base_color

        button_box.draw()
        button_text.draw()
        win.flip()

        if slider.rating is not None and mouse.isPressedIn(button_box):
            intensity = slider.getValue()
            one_trial_data["intensity"] = intensity

            percentage = intensity * 10

            logger.info(f"percentage:{percentage:.2f}%, coresp label: {intensity:.2f}")
            send_marker(lsl_outlet, "RESPONSE_2", is_pre=pre)
            break
    win.setMouseVisible(visibility=False)


def init_encoding_phase():
    global stim_sequence
    if "encoding" not in stim_sequence:
        candidates = positive_words + negative_words
        stim_sequence["encoding"] = generate_trial_sequence(
            n_blocks=1,
            n_trials_per_block=len(candidates),
            stim_list=candidates,
            max_seq_same=3,
        )[0]


def run_encoding_phase():
    send_marker(lsl_outlet, "ENCODING_PHASE_START", is_pre=pre)
    one_trial_data["phase"] = phase_names["encoding"]
    one_trial_data["phase_start_time"] = clock.getTime()

    trials = stim_sequence["encoding"]

    endorse_count = 0

    for idx, trial in enumerate(trials):
        one_trial_data["trial_index"] = idx
        one_trial_data["trial_start_time"] = clock.getTime()
        one_trial_data["stim_word"] = trial
        one_trial_data["stim_type"] = (
            "positive" if trial in positive_words else "negative"
        )

        # 间隔
        draw_fixation(
            get_isi(timing["encoding"]["iti"]["low"], timing["encoding"]["iti"]["high"])
        )

        # Stimulus
        text_stim = visual.TextStim(
            win, text=trial, color="white", height=0.15, font=PSYCHO_FONT
        )
        text_stim.draw()
        win.flip()
        core.wait(timing["encoding"]["stim"])

        draw_fixation(timing["encoding"]["fixation"])

        prompt_stim = visual.TextBox2(
            win,
            text=f"符合我( <c=#51d237>{list(encoding_map.keys())[0].upper()}</c> )   不符合我( <c=#eb5555>{list(encoding_map.keys())[1].upper()}</c> )",
            pos=(0, 0),
            letterHeight=0.08,
            size=(1.2, None),
            font=PSYCHO_FONT,
            color="white",
            alignment="center",
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
            rt = timestamp - onset_time
            one_trial_data["rt"] = rt
            if resp == "yes":
                endorse_count += 1
            logger.info(f"Encoding Trial {idx + 1}: {trial} -> {resp}, rt: {rt:.4f}")
            rating_slider(resp)
            send_marker(lsl_outlet, "RESPONSE", is_pre=pre)
        else:
            send_marker(lsl_outlet, "NO_RESPONSE", is_pre=pre)
        one_trial_data["response"] = resp

        one_trial_data["trial_end_time"] = clock.getTime()
        # 保存该 trial 数据
        update_trial(one_trial_data, one_block_data)
        if pre > 1 and event.getKeys(keyList=["escape"]):
            break
    one_trial_data["endorse_count"] = endorse_count
    one_trial_data["phase_end_time"] = clock.getTime()
    send_marker(lsl_outlet, "ENCODING_PHASE_END", is_pre=pre)

    update_trial(one_trial_data, one_block_data)
    update_block(one_block_data, data_to_save)


def init_exp(config: DictConfig | None):
    global \
        timing, \
        phase_names, \
        stim_sequence, \
        data_to_save, \
        intensity_prompt, \
        intensity_ticks, \
        positive_words, \
        negative_words

    phase_names = config["phase_names"]

    intensity_prompt = config["intensity_prompt"]
    intensity_ticks = list(
        range(config["intensity_ticks"]["start"], config["intensity_ticks"]["end"] + 1)
    )

    if pre or test is False:
        timing = config["timing"]
    if "stim_sequence" in config:
        logger.info("正式实验，使用伪随机序列")

        stim_sequence = config.stim_sequence
        if test:
            import numpy as np

            for k, v in stim_sequence.items():
                np.random.shuffle(v)
                stim_sequence[k] = v[: len(v) // 8]
        positive_words = config.stims["positive"]
        negative_words = config.stims["negative"]

    if "stims" in config:
        if pre:
            total_stims = config.stims["neutral"]
            len_total_stims = len(total_stims)
            positive_words = total_stims[: len_total_stims // 2]
            negative_words = total_stims[len_total_stims // 2 :]

    for key in data_to_save.keys():
        data_to_save[key].clear()


def run_exp(cfg: DictConfig | None):
    if cfg is not None:
        init_exp(cfg)
        title = visual.TextStim(
            win,
            text=cfg.phase_prompt.title,
            color="white",
            height=0.1,
            pos=(0, 0.7),
            font=PSYCHO_FONT,
        )
        prompt = visual.TextBox2(
            win,
            text=cfg.phase_prompt.prompt,
            color="white",
            letterHeight=cfg.phase_prompt.letterHeight or 0.08,
            size=(
                cfg.phase_prompt.size.width or 1.2,
                cfg.phase_prompt.size.height or None,
            ),
            pos=(0, 0),
            font=PSYCHO_FONT,
            alignment=cfg.phase_prompt.alignment or "left",
        )
        title.draw()
        prompt.draw()
        win.flip()
        event.waitKeys(keyList=continue_keys)

    init_encoding_phase()
    run_encoding_phase()


def entry(exp: Experiment | None = None):
    global win, clock, lsl_outlet, logger, pre, test

    win = exp.win or visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")
    clock = exp.clock or core.Clock()
    logger = exp.logger or setup_default_logger()
    lsl_outlet = exp.lsl_outlet or init_lsl("SRETMarker")
    test = exp.test
    # 是否需要预实验
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
    switch_keyboard_layout()
    entry(Experiment(None, None, None, None, None, None))


if __name__ == "__main__":
    main()
