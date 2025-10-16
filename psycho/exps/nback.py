import logging
import random

from omegaconf import DictConfig
from psychopy import core, event, visual
from pylsl import StreamOutlet, local_clock

from psycho.utils import generate_trial_sequence, init_lsl, send_marker, setup_default_logger

# TODO: 是否需要设置 no response
# === 参数设置 ===
nback = 2
n_blocks = 1
n_trials_per_block = 10
stim_pool = list(range(1, 9 + 1))

timing = {
    "fixation": 0.5,
    "stim": 1.0,
    "rest": 1,
}
resp_keys = ["space"]
continue_keys = ["space"]

# === 全局对象 ===
win = None
stim_text = None
clock = None
lsl_outlet = None
logger = None


block_index = 0
trial_index = 0
correct_count = 0  # 正确响应次数
# 存储
stim_sequence = generate_trial_sequence(n_blocks, n_trials_per_block, stim_list=stim_pool)
results = []


def pre_block():
    global correct_count
    # 区块开始前的提示
    text = f"准备进入第 {block_index + 1} 个区块\n你可以按空格键直接开始"

    msg = visual.TextStim(
        win,
        text=text,
        color="white",
        height=0.1,
        wrapWidth=2,
    )
    msg.draw()
    win.flip()
    event.waitKeys(5.0, keyList=continue_keys)
    correct_count = 0


def block():
    global trial_index
    for local_trial_index in range(n_trials_per_block):
        trial_index = local_trial_index
        pre_trial()
        trial()
        post_trial()


def post_block():
    # 区块结束后的提示
    correct_rate = correct_count / n_trials_per_block

    text = f"第 {block_index + 1} 个区块结束\n你共响应了 {correct_count} 次正确响应\n正确率为 {correct_rate * 100:.2f}%\n你有{timing['rest']}秒休息时间\n你可以按空格键进入下一个区块"

    msg = visual.TextStim(
        win,
        text=text,
        color="white",
        height=0.1,
        wrapWidth=2,
    )
    msg.draw()
    win.flip()

    event.waitKeys(timing["rest"], keyList=continue_keys)


def pre_trial():
    fixation = visual.TextStim(win, text="+", color="white", height=0.4, wrapWidth=2)
    fixation.draw()
    win.flip()
    core.wait(timing["fixation"])


def trial():
    global results, correct_count

    stim = stim_sequence[block_index][trial_index]
    stim_text.text = stim
    stim_text.draw()

    # 上横线
    line_top = visual.Line(
        win,
        start=(-0.15, 0.2),
        end=(0.15, 0.2),
        lineColor="white",
        lineWidth=10,
        units="norm",
    )

    # 下横线
    line_bottom = visual.Line(
        win,
        start=(-0.15, -0.2),
        end=(0.15, -0.2),
        lineColor="white",
        lineWidth=10,
        units="norm",
    )

    line_top.draw()
    line_bottom.draw()

    send_marker(lsl_outlet, "TRIAL_START")
    win.flip()
    logger.info(f"Stimulus: {stim}")
    stim_onset = clock.getTime()

    # 等待刺激期间响应
    keys = event.waitKeys(maxWait=timing["stim"], keyList=resp_keys, timeStamped=True)
    response_time = local_clock()

    # 判定 target
    is_target = False
    if trial_index >= nback and stim == stim_sequence[block_index][trial_index - nback]:
        is_target = True

    # 响应情况
    if keys:
        key, rt = keys[0]
        responded = True
        rt = rt - stim_onset
        logger.info(f"Response: {rt}")
    else:
        logger.info("No response")
        responded = False
        rt = None

    # 正确与否
    if is_target and responded:
        correct = True
        correct_count += 1
    elif (not is_target) and (not responded):
        correct = True
        correct_count += 1
    else:
        correct = False

    if correct and responded:
        line_top.color = "green"
        line_bottom.color = "green"
    elif not correct and responded:
        line_top.color = "red"
        line_bottom.color = "red"

    line_top.draw()
    line_bottom.draw()
    stim_text.draw()
    win.flip()
    stim_left = timing["stim"] - (clock.getTime() - stim_onset)
    core.wait(stim_left)

    if is_target:
        if keys:
            send_marker(lsl_outlet, "TARGET_RESPONSE", response_time)
            logger.info("is target and do response")
        else:
            send_marker(lsl_outlet, "TARGET_NORESPONSE", response_time)
            logger.info("is target and no response")
    else:
        if keys:
            send_marker(lsl_outlet, "NOT_TARGET_RESPONSE", response_time)
            logger.info("is not target and do response")
        else:
            send_marker(lsl_outlet, "NOT_TARGET_NORESPONSE", response_time)
            logger.info("is not target and no response")

    results.append([trial_index, stim, is_target, responded, rt, correct])


def post_trial():
    win.flip()
    core.wait(0.5)


def init_exp(config: DictConfig | None):
    global nback, n_blocks, n_trials_per_block, timing, stim_sequence

    n_blocks = config.n_blocks
    n_trials_per_block = config.n_trials_per_block
    timing = config.timing
    nback = config.nback
    if "stim_sequence" in config:
        stim_sequence = config.stim_sequence


def run_exp(cfg: DictConfig | None):
    """运行实验"""
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
    lsl_outlet_session: StreamOutlet | None = None,
    config: DictConfig | None = None,
    logger_session: logging.Logger | None = None,
):
    """实验入口"""
    global stim_text, lsl_outlet, win, clock, logger
    win = win_session if win_session else visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")

    clock = clock_session if clock_session else core.Clock()
    logger = logger_session if logger_session else setup_default_logger()

    lsl_outlet = lsl_outlet_session if lsl_outlet_session else init_lsl("NBackMarker")  # 初始化 LSL
    stim_text = visual.TextStim(win, text="", color="white", height=0.3, wrapWidth=2)

    if config is not None and "pre" in config:
        while True:
            run_exp(config.pre)

            commit_text = "是否需要再次进行预实验?\n按 y 键再次进行预实验, 按 n 键结束预实验"
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

    send_marker(lsl_outlet, "EXPERIMENT_START")
    logger.info("实验开始")
    run_exp(config.full if config is not None else None)
    # 实验结束
    send_marker(lsl_outlet, "EXPERIMENT_END")
    logger.info("实验结束")


def main():
    entry()


if __name__ == "__main__":
    main()
