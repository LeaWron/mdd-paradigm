import random

from psychopy import core, event, visual

from psycho.utils import adapt_image_stim_size, init_lsl, parse_stim_path, send_marker

# === 参数设置 ===
n_blocks = 1
n_trials_per_block = 10
continue_keys = ["space"]

fixation_duration = 0.5

stim_folder = parse_stim_path("emotion-face")
stim_items = list(stim_folder.glob("*"))
stim_duration = 0.5

response_map = {"left": "positive", "down": "neural", "right": "negative"}
max_response_time = 2

rest_duration = 30
# === 全局变量 ===
win = None
clock = None
lsl_outlet = None
block_index = 0


def pre_block():
    text = f"当前 block 为第 {block_index + 1} 个 block, 请按空格键开始"
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
    text = f"你有 {rest_duration} 秒休息时间, 或者你可以选择按空格键进入下一个 block"
    text_stim = visual.TextStim(win, text=text, color="white", wrapWidth=2)
    text_stim.draw()
    win.flip()
    event.waitKeys(rest_duration, keyList=continue_keys)


def pre_trial():
    fixation = visual.TextStim(win, text="+", color="white", height=0.4, wrapWidth=2)
    fixation.draw()
    win.flip()
    core.wait(fixation_duration)


def trial():
    def get_stim():
        stim_item = random.choice(stim_items)
        if stim_item.name.startswith("p"):
            label = "positive"
        elif stim_item.name.startswith("n"):
            label = "negative"
        else:
            label = "neural"
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

    core.wait(stim_duration)

    keys = event.waitKeys(maxWait=max_response_time, keyList=list(response_map.keys()), timeStamped=True)

    if keys:
        key, rt = keys[0]

        resp_emotion = response_map.get(key, None)
        correct = resp_emotion == label
        send_marker(
            lsl_outlet,
            f"{label}_{resp_emotion}_{correct}",
            rt,
        )


def post_trial():
    # TODO: 是否需要 rate intensity
    win.flip()
    core.wait(0.5)


def entry(
    win_session: visual.Window | None = None,
    clock_session: core.Clock | None = None,
):
    global win, clock, lsl_outlet, block_index
    win = win_session if win_session else visual.Window(fullscr=True, color="grey", units="norm")
    clock = clock_session if clock_session else core.Clock()

    lsl_outlet = init_lsl("emotion_face")

    for local_block_index in range(n_blocks):
        block_index = local_block_index
        pre_block()
        block()
        post_block()
    send_marker(
        lsl_outlet,
        "EXPERIMENT_END",
        clock.getTime(),
    )


def main():
    entry()


if __name__ == "__main__":
    main()
