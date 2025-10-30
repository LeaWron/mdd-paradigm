from pathlib import Path

from omegaconf import DictConfig
from psychopy import core, event, visual

from psycho.session import Experiment
from psycho.utils import (
    adapt_image_stim_size,
    generate_trial_sequence,
    init_lsl,
    into_stim_str,
    parse_stim_path,
    save_csv_data,
    send_marker,
    setup_default_logger,
    update_block,
    update_trial,
)

# === 参数设置 ===
n_blocks = 1
n_trials_per_block = 5
continue_keys = ["space"]

timing = {
    "fixation": 0.5,
    "stim": 1.0,
    "max_response": 2.0,
    "emotion_select": 1.0,
    "rest": 20.0,
}

stim_folder = parse_stim_path("emotion-face")
stim_items = list(stim_folder.glob("*.BMP"))

response_map = {"left": "positive", "down": "neutral", "right": "negative"}


intensity_prompt = "请选择情感的强度（1-9）, 1为最弱，9为最强"
intensity_ticks = list(range(1, 10))
intensity_tips = ["最弱", "中等", "最强"]
# === 全局变量 ===
win = None
clock = None
lsl_outlet = None
logger = None

block_index = 0
trial_index = 0

correct_count = 0
pre = False


stim_sequence = generate_trial_sequence(
    n_blocks,
    n_trials_per_block,
    max_seq_same=3,
    stim_list=stim_items,
    stim_weights=None,
)

for block, stims in stim_sequence.items():
    block_seq = []
    for stim_path in stims:
        stim_str = into_stim_str(stim_path)
        block_seq.append({"stim_path": stim_str, "label": 0})
    stim_sequence[block] = block_seq


# === 数据保存 ===
data_to_save = {
    "exp_start_time": [],
    "exp_end_time": [],
    "block_index": [],
    "trial_index": [],
    "trial_start_time": [],
    "trial_end_time": [],
    "stim": [],
    "label_intensity": [],
    "choice": [],
    "rt": [],
    "intensity": [],
    "correct_rate": [],
}

one_trial_data = {key: None for key in data_to_save.keys()}
one_block_data = {key: [] for key in data_to_save.keys()}


def pre_block():
    text = f"当前 block 为第 {block_index + 1} 个 block\n记住左方向键为积极, 上方向键为中性, 右方向键为消极\n请按空格键开始"
    text_stim = visual.TextStim(win, text=text, color="white", wrapWidth=2)
    text_stim.draw()
    win.flip()
    event.waitKeys(keyList=continue_keys)


def block():
    global trial_index
    for local_trial_index in range(n_trials_per_block):
        trial_index = local_trial_index
        one_trial_data["trial_index"] = trial_index
        one_trial_data["trial_start_time"] = clock.getTime()

        pre_trial()
        trial()
        post_trial()

        one_trial_data["trial_end_time"] = clock.getTime()
        update_trial(one_trial_data, one_block_data)


def post_block():
    correct_rate = 1.0 * correct_count / n_trials_per_block

    one_trial_data["correct_rate"] = correct_rate
    # resting
    text = f"你有 {timing['rest']} 秒休息时间\n你可以按空格键进入下一个 block"
    text_stim = visual.TextStim(win, text=text, color="white", wrapWidth=2)
    text_stim.draw()
    win.flip()
    event.waitKeys(timing["rest"], keyList=continue_keys)


def pre_trial():
    fixation = visual.TextStim(win, text="+", color="white", height=0.4, wrapWidth=2)
    fixation.draw()
    win.flip()
    core.wait(timing["fixation"])


def trial():
    global correct_count

    def get_stim_kind(stim_item_path: Path):
        stim_item = parse_stim_path(stim_item_path)
        seq = int(stim_item.stem.rsplit("-", 1)[-1])
        if seq < 10:
            kind_label = "positive"
        elif seq > 11:
            kind_label = "negative"
        else:
            kind_label = "neutral"
        return stim_item, kind_label

    stim_path: str = stim_sequence[block_index][trial_index]["stim_path"]
    stim_item, kind_label = get_stim_kind(stim_path)

    one_trial_data["stim"] = kind_label
    one_trial_data["label_intensity"] = stim_sequence[block_index][trial_index]["label"]

    stim_height, aspect_ratio = adapt_image_stim_size(stim_item, 1)

    stimulus = visual.ImageStim(
        win,
        image=stim_item,
        size=(stim_height * aspect_ratio, stim_height),
    )
    stimulus.draw()
    win.flip()
    send_marker(
        lsl_outlet,
        "TRIAL_START",
        is_pre=pre,
    )

    core.wait(timing["stim"])
    judge_stim = visual.TextStim(
        win, text="请判断这张图片中的人脸的情绪类别", color="white", wrapWidth=2
    )
    judge_stim.draw()
    win.flip()
    on_set = clock.getTime()

    keys = event.waitKeys(
        maxWait=timing["max_response"],
        keyList=list(response_map.keys()),
        timeStamped=True,
    )
    resp_emotion = None

    if keys:
        key, rt = keys[0]
        rt -= on_set

        resp_emotion = response_map.get(key, None)
        correct = resp_emotion == kind_label
        if correct:
            correct_count += 1
        one_trial_data["choice"] = resp_emotion
        one_trial_data["rt"] = rt

        send_marker(lsl_outlet, "RESPONSE_1", is_pre=pre)

        logger.info(f"correct: {correct}, resp_emotion: {resp_emotion}, rt: {rt}")
    else:
        send_marker(lsl_outlet, "NORESPONSE", is_pre=pre)

        logger.info("NO RESPONSE")

    prompt, slider = rating_slider()

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
        win, text="确认", color="white", pos=(0, -0.5), height=0.1
    )

    mouse = event.Mouse(win=win)
    # 选择中性的时候不打分
    while resp_emotion != "neutral" and True:
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

            logger.info(f"intensity: {intensity}")
            send_marker(lsl_outlet, "RESPONSE_2", is_pre=pre)
            break

    # 选择了中性情绪, 则强度为0
    if resp_emotion == "neutral":
        one_trial_data["intensity"] = 0
        send_marker(lsl_outlet, "RESPONSE_3", is_pre=pre)


def post_trial():
    win.flip()
    core.wait(0.5)


def rating_slider():
    prompt = visual.TextStim(
        win, text=intensity_prompt, color="white", pos=(0, 0.4), wrapWidth=2
    )
    slider = visual.Slider(
        win,
        ticks=intensity_ticks,
        labels=intensity_tips,
        granularity=0,  # 连续可拖动
        # startValue=labels[len(labels) // 2],
        size=(0.9, 0.05),
        style=["rating"],
        color="white",
        pos=(0, 0),
        font="SimSun",
    )
    # slider.setValue(slider.startValue)
    return prompt, slider


def init_exp(config: DictConfig | None = None):
    global \
        n_blocks, \
        n_trials_per_block, \
        timing, \
        response_map, \
        intensity_prompt, \
        intensity_ticks, \
        intensity_tips, \
        stim_sequence

    n_blocks = config.n_blocks
    n_trials_per_block = config.n_trials_per_block
    timing = config.timing

    response_map = config.response_map
    intensity_prompt = config.intensity_prompt
    intensity_ticks = list(
        range(config.intensity_ticks.start, config.intensity_ticks.end + 1)
    )
    intensity_tips = config.intensity_tips

    if "stim_sequence" in config:
        stim_sequence = config.stim_sequence


def run_exp(cfg: DictConfig | None):
    global block_index

    if cfg is not None:
        init_exp(cfg)
        prompt = visual.TextStim(
            win,
            text=cfg.phase_prompt,
            color="white",
            height=0.06,
            wrapWidth=2,
        )
        prompt.draw()
        win.flip()
        event.waitKeys(keyList=continue_keys)

    for local_block_index in range(n_blocks):
        block_index = local_block_index
        one_trial_data["block_index"] = block_index

        pre_block()
        block()
        post_block()

        update_block(one_block_data, data_to_save)


def entry(exp: Experiment | None = None):
    global win, clock, lsl_outlet, logger, pre
    win = exp.win or visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")

    clock = exp.clock or core.Clock()
    logger = exp.logger or setup_default_logger()

    lsl_outlet = exp.lsl_outlet or init_lsl("EmotionFaceMarker")  # 初始化 LSL

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

    one_trial_data["exp_start_time"] = clock.getTime()
    send_marker(lsl_outlet, "EXPERIMENT_START", is_pre=pre)
    logger.info("实验开始")

    run_exp(exp.config.full if exp.config is not None else None)

    send_marker(lsl_outlet, "EXPERIMENT_END", is_pre=pre)
    logger.info("实验结束")
    one_trial_data["exp_end_time"] = clock.getTime()

    if exp.config is not None:
        logger.info("保存数据")

        update_trial(one_trial_data, one_block_data)
        update_block(one_block_data, data_to_save)
        save_csv_data(data_to_save, exp.session_info["save_path"] + "-face_recognition")


def main():
    entry()


if __name__ == "__main__":
    main()
