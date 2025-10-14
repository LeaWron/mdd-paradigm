from omegaconf import DictConfig
from psychopy import core, event, prefs, sound, visual
from pylsl import StreamOutlet

from psycho.utils import init_lsl, parse_stim_path, send_marker

# === 偏好设置 ===
prefs.hardware["audioDevice"] = "扬声器 (2- High Definition Audio Device)"

# === 参数设置 ===
n_blocks = 2
phase = ["eye_close", "eye_open"]
timing = {
    "max_wait_duration": 1 * 1,  # 等待时间，单位秒
    "resting_duration": 1 * 5,  # 休息时间，单位秒
}
block_cfg = {
    "eye_close": {
        "prompt": "请闭眼, 直到再次听到如下提示音",  # 闭眼
        "marker": "EYE_CLOSE",
    },
    "eye_open": {
        "prompt": "请睁眼, 直到再次听到如下提示音",  # 睁眼
        "marker": "EYE_OPEN",
    },
}


continue_keys = ["space"]
notification = parse_stim_path("notification.wav")


# === 全局参数 ===
win = None
clock = None
lsl_outlet = None
block_index = 0


def pre_block():
    text = block_cfg[phase[block_index]]["prompt"]
    stim = visual.TextStim(win, text=text, color="white", height=0.1, wrapWidth=2)
    stim.draw()
    win.flip()
    core.wait(0.5)
    sound_prompt = sound.Sound(notification, secs=1)
    sound_prompt.play()
    event.waitKeys(timing["max_wait_duration"], keyList=continue_keys)


def block():
    send_marker(lsl_outlet, block_cfg[phase[block_index]]["marker"])
    text = block_cfg[phase[block_index]]["prompt"]
    stim = visual.TextStim(win, text=text, color="white", height=0.1, wrapWidth=2)
    stim.draw()
    win.flip()
    core.wait(timing["resting_duration"])
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


def init_exp(config: DictConfig | None = None):
    def read_config(cfg: DictConfig):
        global n_blocks, timing, phase, block_cfg, notification
        if cfg is not None:
            n_blocks = cfg.n_blocks
            timing = cfg.timing
            phase = cfg.phase
            block_cfg = cfg
            notification = parse_stim_path(cfg.notification)

    if config is not None:
        read_config(config)


def run_exp(cfg: DictConfig | None):
    global block_index

    init_exp(cfg)
    for local_block_index in range(n_blocks):
        block_index = local_block_index
        pre_block()
        block()
        post_block()


def entry(
    win_session: visual.Window | None = None,
    lsl_outlet_session: StreamOutlet | None = None,
    clock_session: core.Clock | None = None,
    config: DictConfig | None = None,
):
    global win, lsl_outlet, clock
    win = win_session or visual.Window(fullscr=True, color="grey", units="norm")
    lsl_outlet = lsl_outlet_session or init_lsl("RestingStateMarker")
    clock = clock_session or core.Clock()

    # 预实验
    if config is not None and "pre" in config:
        run_exp(config.pre)

    # 正式实验
    send_marker(lsl_outlet, "EXPERIMENT_START")
    run_exp(None if config is None else config.full)
    send_marker(lsl_outlet, "EXPERIMENT_END")


def main():
    entry()


if __name__ == "__main__":
    main()
