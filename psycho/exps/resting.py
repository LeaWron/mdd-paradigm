from psychopy import core, event, prefs, sound, visual
from pylsl import StreamOutlet

from psycho.utils import init_lsl, parse_stim_path, send_marker

# === 偏好设置 ===
prefs.hardware["audioDevice"] = "扬声器 (2- High Definition Audio Device)"
# === 参数设置 ===
n_blocks = 2
adaption_duration = 1 * 1  # 适应时间，单位秒
resting_duration = 1 * 5  # 休息时间，单位秒
# TODO：添加提示文本
prompts = [
    "请闭眼, 直到再次听到如下提示音",  # 闭眼
    "请睁眼, 直到再次听到如下提示音",  # 睁眼
]
markers = [
    "CLOSE_EYES",
    "OPEN_EYES",
]

continue_keys = ["space"]
notification = parse_stim_path("notification.wav")


# === 全局参数 ===
win = None
clock = None
lsl_outlet = None
block_index = 0


def pre_block():
    text = prompts[block_index]
    stim = visual.TextStim(win, text=text, color="white", height=0.1, wrapWidth=2)
    stim.draw()
    win.flip()
    core.wait(0.5)
    sound_prompt = sound.Sound(notification, secs=1)
    sound_prompt.play()
    event.waitKeys(adaption_duration, keyList=continue_keys)


def block():
    send_marker(lsl_outlet, markers[block_index])
    text = prompts[block_index]
    stim = visual.TextStim(win, text=text, color="white", height=0.1, wrapWidth=2)
    stim.draw()
    win.flip()
    core.wait(resting_duration)
    sound_prompt = sound.Sound(notification, secs=1)
    sound_prompt.play()
    core.wait(1)


def post_block():
    fixation = visual.TextStim(win, text="+", color="white", height=0.1, wrapWidth=2)
    fixation.draw()
    win.flip()
    core.wait(0.5)


def pre_trial():
    pass


def trial():
    pass


def post_trial():
    pass


def entry(
    win_session: visual.Window | None = None,
    lsl_outlet_session: StreamOutlet | None = None,
    clock_session: core.Clock | None = None,
):
    global win, lsl_outlet, clock, block_index
    win = win_session or visual.Window(fullscr=True, color="grey", units="norm")
    lsl_outlet = lsl_outlet_session or init_lsl("RestingStateMarker")
    clock = clock_session or core.Clock()

    send_marker(lsl_outlet, "EXPERIMENT_START")
    for local_block_index in range(n_blocks):
        block_index = local_block_index
        pre_block()
        block()
        post_block()
    send_marker(lsl_outlet, "EXPERIMENT_END")


def main():
    entry()


if __name__ == "__main__":
    main()
