from pathlib import Path

from omegaconf import DictConfig
from psychopy import core, event, visual

from GazeFollower.example.MyCalibrationPsycho import eyetracking_calibration
from psycho.session import PSYCHO_FONT, Experiment
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
n_blocks = 2
n_trials_per_block = 15
continue_keys = ["space"]

timing = {
    "fixation": 0.5,
    "stim": 1.0,
    "max_response": 2.0,
    "emotion_select": 1.0,
    "rest": 20,
}

stim_folder = parse_stim_path("emotion-face")
stim_items = list(stim_folder.glob("*.BMP"))

response_map = {"a": "positive", "s": "neutral", "d": "negative"}


intensity_prompt = (
    "请选择情感的强度（1-9）, 1为<c=#eb5555>最弱</c>, 9为<c=#51d237>最强</c>"
)
intensity_ticks = list(range(1, 10))
# === 全局变量 ===
win = None
clock = None
lsl_outlet = None
logger = None

block_index = 0
trial_index = 0

correct_count = 0
pre = 0
test = False


stim_sequence = generate_trial_sequence(
    n_blocks,
    n_trials_per_block,
    max_seq_same=1,
    all_occur=True,
    stim_list=stim_items,
    stim_weights=None,
)

for block, stims in stim_sequence.items():
    block_seq = []
    for stim_path in stims:
        stim_str = into_stim_str(stim_path)
        label = float(stim_str.rsplit("-")[-2])
        block_seq.append({"stim_path": stim_str, "label": label})
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
    text_front = f"当前区块为第 {block_index + 1} 个区块\n" if not pre else ""
    text = f"{text_front}记住按 <c=#51d237>A</c> 为<c=yellow>积极</c>, <c=#51d237>S</c> 为<c=white>中性</c>, <c=#51d237>D</c> 为<c=purple>消极</c>\n请按<c=#51d237>空格键</c>开始"
    text_stim = visual.TextBox2(
        win,
        text=text,
        color="white",
        letterHeight=0.08,
        size=(1.2, None),
        font=PSYCHO_FONT,
        alignment="center",
    )

    text_stim.draw()
    win.flip()
    event.waitKeys(keyList=continue_keys)


def block():
    global trial_index
    for local_trial_index in range(n_trials_per_block):
        trial_index = local_trial_index
        one_trial_data["trial_index"] = trial_index
        one_trial_data["trial_start_time"] = clock.getTime()

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

        one_trial_data["trial_end_time"] = clock.getTime()
        update_trial(one_trial_data, one_block_data)


def post_block():
    global correct_count
    correct_rate = 1.0 * correct_count / n_trials_per_block
    correct_count = 0

    one_trial_data["correct_rate"] = correct_rate
    logger.info(
        f"Block {block_index + 1} end, current block correct rate: {correct_rate}"
    )
    update_trial(one_trial_data, one_block_data)
    # resting
    text_front = "" if pre else "该区块结束\n"
    for i in range(timing["rest"], -1, -1):
        text = f"{text_front}你有 <c=yellow>{i}</c> 秒休息时间\n你可以按<c=#51d237>空格键</c>继续"
        text_stim = visual.TextBox2(
            win,
            text=text,
            color="white",
            letterHeight=0.08,
            size=(1.2, None),
            font=PSYCHO_FONT,
            alignment="center",
        )
        text_stim.draw()
        win.flip()

        if event.waitKeys(1, keyList=continue_keys):
            break


def pre_trial():
    fixation = visual.TextStim(
        win, text="+", color="white", height=0.2, wrapWidth=2, font=PSYCHO_FONT
    )
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

    stim_height, aspect_ratio = adapt_image_stim_size(win, stim_item, 1)

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
        win,
        text="请判断",
        color="white",
        wrapWidth=2,
        font=PSYCHO_FONT,
    )
    judge_stim.draw()
    judge_prompt = visual.TextBox2(
        win,
        text="<c=yellow>积极</c>( <c=#51d237>A</c> )  <c=white>中性</c>( <c=#51d237>S</c> )  <c=purple>消极</c>( <c=#51d237>D</c> )",
        color="white",
        pos=(0, -0.3),
        letterHeight=0.08,
        size=(1.2, None),
        font=PSYCHO_FONT,
        alignment="center",
    )
    judge_prompt.draw()
    win.flip()
    on_set = clock.getTime()

    keys = event.waitKeys(
        maxWait=timing["max_response"],
        keyList=list(response_map.keys()),
        timeStamped=clock,
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

        logger.info(
            f"Block {block_index + 1}, trial {trial_index + 1}: correct_emotion: {one_trial_data['stim']}, resp_emotion: {resp_emotion}, rt: {rt:.4f}"
        )
    else:
        send_marker(lsl_outlet, "NORESPONSE", is_pre=pre)

        logger.info("NO RESPONSE")
        visual.TextStim(
            win,
            text="你没有及时进行评价, 请集中精神",
            color="white",
            wrapWidth=2,
            font=PSYCHO_FONT,
        ).draw()
        win.flip()
        core.wait(1.0)
        return

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
        win,
        text="确认",
        color="white",
        pos=(0, -0.5),
        height=0.1,
        font=PSYCHO_FONT,
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

            logger.info(
                f"true_intensity: {one_trial_data['label_intensity']:.2f}, selected_intensity: {intensity:.2f}"
            )
            send_marker(lsl_outlet, "RESPONSE_2", is_pre=pre)
            break

    # 选择了中性情绪, 则强度为0
    if resp_emotion == "neutral":
        one_trial_data["intensity"] = 0
        logger.info(
            f"neutral, while true intensity is {one_trial_data['label_intensity']:.2f}"
        )
        send_marker(lsl_outlet, "RESPONSE_3", is_pre=pre)


def post_trial():
    win.setMouseVisible(visibility=False)
    win.flip()
    core.wait(0.5)


def rating_slider():
    win.setMouseVisible(visibility=True)
    prompt = visual.TextBox2(
        win,
        text=intensity_prompt,
        color="white",
        pos=(0, 0.4),
        size=(2, None),
        letterHeight=0.08,
        font=PSYCHO_FONT,
        alignment="center",
    )
    slider = visual.Slider(
        win,
        ticks=intensity_ticks,
        labels=intensity_ticks,
        granularity=0,  # 连续可拖动
        # startValue=labels[len(labels) // 2],
        size=(0.9, 0.05),
        style=["rating"],
        color="white",
        pos=(0, 0),
        font=PSYCHO_FONT,
        labelHeight=0.08,
    )
    # slider.setValue(slider.startValue)
    return prompt, slider


def show_stims():
    current_stim_sequence = stim_sequence[block_index]
    current_most = set()
    for stimulus in current_stim_sequence:
        if stimulus["label"] == 9:
            current_most.add(parse_stim_path(stimulus["stim_path"]))
    current_most_stim = sorted(list(current_most))
    for i, stim_path in enumerate(current_most_stim):
        height, aspect_ratio = adapt_image_stim_size(win, stim_path, 0.5)
        stimulus = visual.ImageStim(
            win,
            image=stim_path,
            pos=(-0.5 + i, 0),
            size=(height * aspect_ratio, height),
        )
        stimulus.draw()
    visual.TextBox2(
        win,
        text="这里为你展示当前区块中积极和消极图片中情感强度最高的图片",
        color="white",
        pos=(0, 0.5),
        size=(2, None),
        letterHeight=0.08,
        alignment="center",
        font=PSYCHO_FONT,
    ).draw()
    visual.TextBox2(
        win,
        text="请按<c=#51d237>空格键</c>继续",
        color="white",
        pos=(0, -0.7),
        letterHeight=0.08,
        alignment="center",
        size=(2, None),
        font=PSYCHO_FONT,
    ).draw()
    win.flip()

    event.waitKeys(keyList=continue_keys)


def init_exp(config: DictConfig | None = None):
    global \
        n_blocks, \
        n_trials_per_block, \
        timing, \
        response_map, \
        intensity_prompt, \
        intensity_ticks, \
        stim_sequence

    if pre or not test:
        logger.info("Run in real exp")
        n_blocks = config.n_blocks
        n_trials_per_block = config.n_trials_per_block
        timing = config.timing

    response_map = config.response_map
    intensity_prompt = config.intensity_prompt
    intensity_ticks = list(
        range(config.intensity_ticks.start, config.intensity_ticks.end + 1)
    )

    if "stim_sequence" in config:
        stim_sequence = config.stim_sequence
    for k in data_to_save.keys():
        data_to_save[k].clear()


def run_exp(cfg: DictConfig | None):
    global block_index

    if cfg is not None:
        init_exp(cfg)
        title = visual.TextStim(
            win,
            text=cfg.phase_prompt.title,
            color="white",
            height=0.08,
            pos=(0, 0.7),
            font=PSYCHO_FONT,
        )
        prompt = visual.TextBox2(
            win,
            text=cfg.phase_prompt.prompt,
            color="white",
            letterHeight=cfg.phase_prompt.letterHeight or 0.06,
            size=(
                cfg.phase_prompt.size.width or 1.5,
                cfg.phase_prompt.size.height or None,
            ),
            pos=(0, 0),
            alignment=cfg.phase_prompt.alignment or "left",
            font=PSYCHO_FONT,
        )
        title.draw()
        prompt.draw()
        win.flip()
        event.waitKeys(keyList=continue_keys)

    for local_block_index in range(n_blocks):
        block_index = local_block_index
        one_trial_data["block_index"] = block_index
        if not pre:
            show_stims()

        pre_block()
        block()
        post_block()

        update_block(one_block_data, data_to_save)


def entry(exp: Experiment | None = None):
    global win, clock, lsl_outlet, logger, pre, test
    win = exp.win or visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")

    clock = exp.clock or core.Clock()
    logger = exp.logger or setup_default_logger()

    lsl_outlet = exp.lsl_outlet or init_lsl("EmotionFaceMarker")  # 初始化 LSL
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
    # eyetracking calibration
    # [ ] 预实验后校准吗?
    # [ ] 讲解更详细?或者人来讲解?
    if exp.camera is not None:
        calibration_text = "下面进行眼动校准,随后的预览中请确保您的面部完整的呈现在画面中\n按<c=#51d237>空格键</c>继续"
        calibration_info = visual.TextBox2(
            win,
            text=calibration_text,
            color="white",
            letterHeight=0.08,
            size=(2, None),
            alignment="center",
            pos=(0, 0),
            font=PSYCHO_FONT,
        )
        calibration_info.draw()
        win.flip()
        event.waitKeys(keyList=continue_keys)
        logger.info("begin eyetracking calibration")

        def calibrate_eyetracking():
            ret = eyetracking_calibration(
                win=win, camera=exp.camera, formal=True, info=exp.session_info
            )
            if ret == 0:
                logger.info("eyetracking failed")
            else:
                logger.info("eyetracking calibration succeeded")
            return ret

        send_marker(
            lsl_outlet=lsl_outlet, marker="EYETRACKING_CALIBRATION_START", is_pre=pre
        )
        calibrate_eyetracking()
        send_marker(
            lsl_outlet=lsl_outlet, marker="EYETRACKING_CALIBRATION_END", is_pre=pre
        )

        logger.info("calibration done")

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
    entry(Experiment(None, None, None, None, None, session_info=None, camera=None))


if __name__ == "__main__":
    main()
