import random

from omegaconf import DictConfig
from psychopy import core, event, visual
from pylsl import StreamOutlet

from psycho.utils import adapt_image_stim_size, init_lsl, parse_stim_path, send_marker

# === 参数设置 ===
n_blocks = 1
n_trials_per_block = 10
continue_keys = ["space"]

timing = {
    "fixation": 0.5,
    "stim": 1.0,
    "max_response": 2.0,
    "emotion_select": 1.0,
    "rest": 20.0,
}

stim_folder = parse_stim_path("emotion-face")
stim_items = list(stim_folder.glob("*"))

response_map = {"left": "positive", "right": "negative"}

emotion_labels = {
    "快乐": "happy",
    "悲伤": "sad",
    "愤怒": "angry",
    "恐惧": "fear",
    "厌恶": "disgust",
    "中性": "neutral",
}
# === 全局变量 ===
win = None
clock = None
lsl_outlet = None
block_index = 0
port = None


def pre_block():
    text = f"当前 block 为第 {block_index + 1} 个 block\n记住左方向键为积极, 右方向键为消极\n请按空格键开始"
    text_stim = visual.TextStim(win, text=text, color="white", wrapWidth=2)
    text_stim.draw()
    win.flip()
    event.waitKeys(keyList=continue_keys)


def block():
    for _ in range(n_trials_per_block):
        pre_trial()
        trial()
        post_trial()


def post_block():
    # resting
    text = f"你有 {timing['rest']} 秒休息时间, 或者你可以选择按空格键进入下一个 block"
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
    def get_stim():
        stim_item = random.choice(stim_items)
        if stim_item.name.startswith("p"):
            label = "positive"
        elif stim_item.name.startswith("n"):
            label = "negative"
        else:
            label = "neutral"
        return stim_item, label

    stim_item, label = get_stim()
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
    )

    core.wait(timing["stim"])

    keys = event.waitKeys(maxWait=timing["max_response"], keyList=list(response_map.keys()), timeStamped=True)

    if keys:
        key, rt = keys[0]

        resp_emotion = response_map.get(key, None)
        correct = resp_emotion == label
        send_marker(lsl_outlet, "RESPONSE_1")


def post_trial():
    win.flip()

    def emotion_label_select():
        # 显示情感标签选择界面
        prompt = visual.TextStim(
            win,
            text="请选择情感标签:快乐、悲伤、愤怒、恐惧、厌恶、中性\n使用鼠标选择标签, 按空格键确认",
            color="white",
            height=0.06,
            pos=(0, 0.3),
            wrapWidth=2,
        )
        emotion_select = visual.Slider(
            win,
            pos=(0, 0),
            # size=(1, 0.05),
            ticks=None,
            labels=list(emotion_labels.keys()),
            font="SimSun",
            granularity=1,
            color="white",
            style="radio",
        )
        while True:
            prompt.draw()
            emotion_select.draw()
            win.flip()

            keys = event.getKeys(keyList=continue_keys)
            if keys and emotion_select.getRating() is not None:
                selected_emtion_key = emotion_select.getRating()
                selected_emotion_label = emotion_labels[selected_emtion_key]
                print(f"选择了情感标签: {selected_emotion_label}")
                break

        send_marker(lsl_outlet, "RESPONSE_2")

    emotion_label_select()
    core.wait(0.5)


def init_exp(config: DictConfig | None = None):
    global n_blocks, n_trials_per_block, timing, go_prob

    n_blocks = config.n_blocks
    n_trials_per_block = config.n_trials_per_block
    timing = config.timing
    go_prob = config.go_prob


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
    lsl_outlet_session: StreamOutlet = None,
    config: DictConfig | None = None,
):
    global win, clock, lsl_outlet, block_index
    win = win_session if win_session else visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")

    clock = clock_session if clock_session else core.Clock()

    lsl_outlet = lsl_outlet_session if lsl_outlet_session else init_lsl("EmotionFaceMarker")  # 初始化 LSL

    if config is not None and "pre" in config:
        run_exp(config.pre)

    send_marker(lsl_outlet, "EXPERIMENT_START")
    run_exp(config.full if config is not None else None)
    send_marker(lsl_outlet, "EXPERIMENT_END")


def main():
    entry()


if __name__ == "__main__":
    main()
