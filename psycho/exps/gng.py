import logging

from omegaconf import DictConfig
from psychopy import core, event, visual
from pylsl import StreamOutlet

from psycho.utils import arbitary_keys, generate_trial_sequence, get_isi, init_lsl, send_marker, setup_default_logger

# TODO: 缩短 trial 时间
# === 参数设置 ===
n_blocks = 1  # block 数量
n_trials_per_block = 10  # 每个 block 的 trial 数
resp_keys = ["space"]  # 受试者按键
continue_keys = ["space"]

timing = {
    "fixation": 0.5,
    "total_trial": 2.5,
    "rest": 30,
    "iti": {
        "lower": 0.5,
        "upper": 1.0,
    },
}

go_prob = 0.7  # Go trial 的概率
stim_list = [True, False]
stim_weights = [go_prob, 1 - go_prob]
max_seq_same = 4  # 最大连续相同 trial 数

# === 全局变量 ===
block_index = 0
trial_index = 0
win = None  # 全局窗口对象
clock = None  # 全局时钟对象
lsl_outlet = None
logger = None  # 全局 logger 对象

stim_sequence = generate_trial_sequence(
    n_blocks,
    n_trials_per_block,
    max_seq_same=max_seq_same,
    stim_list=stim_list,
    stim_weights=stim_weights,
)


# 实验部分
def pre_block():
    text = f"准备进入第 {block_index + 1} 个区块\n你可以按空格键开始"

    msg = visual.TextStim(
        win,
        color="white",
        text=text,
        height=0.1,
        wrapWidth=2,
        anchorHoriz="center",
        anchorVert="center",
    )
    msg.draw()
    win.flip()
    event.waitKeys(timing["rest"] if block_index > 0 else 5.0, keyList=arbitary_keys)

    # send_marker(lsl_outlet, f"BLOCK_START_{block_index}")


def block():
    global trial_index
    for local_trial_index in range(n_trials_per_block):
        trial_index = local_trial_index
        pre_trial()
        trial()
        post_trial()
    # send_marker(lsl_outlet, f"BLOCK_END_{block_index}")


def post_block():
    msg = visual.TextStim(
        win,
        text=f"第 {block_index + 1} 个区块结束\n你有{timing['rest']}秒休息时间\n按任意键继续",
        color="white",
        height=0.1,
        wrapWidth=2,
    )
    msg.draw()
    win.flip()
    event.waitKeys(5.0, keyList=arbitary_keys)


def pre_trial():
    # 空屏 + 注视点
    fixation = visual.TextStim(win, text="+", color="white", height=0.4, wrapWidth=2)
    fixation.draw()
    win.flip()
    core.wait(timing["fixation"])


def trial():
    is_go = stim_sequence[block_index][trial_index]

    win.flip()
    blank_duration = get_isi(timing["iti"]["lower"], timing["iti"]["upper"])
    core.wait(blank_duration)

    stim_color = "#23af27" if is_go else "#d43a3a"
    stim = visual.Circle(
        win,
        radius=0.2,
        color=stim_color,
        colorSpace="hex",
    )
    stim.draw()

    send_marker(lsl_outlet, "TRIAL_START")
    logger.info(f"当前 trial: {trial_index}, is_go: {is_go}")
    win.flip()

    # trial 开始 marker
    # 反应
    keys = event.waitKeys(
        maxWait=timing["total_trial"] - blank_duration,
        keyList=resp_keys,
        timeStamped=True,
    )
    # 反应 marker
    if is_go:
        if keys:
            logger.info(f"GO trial 反应时间: {keys[0][1]}")
            send_marker(lsl_outlet, "GO_RESPONSE")
        else:
            logger.info("GO trial 未反应")
            send_marker(lsl_outlet, "GO_NORESPONSE")
    else:
        if keys:
            logger.info(f"NOGO trial 反应时间: {keys[0][1]}")
            send_marker(lsl_outlet, "NOGO_RESPONSE")
        else:
            logger.info("NOGO trial 未反应")
            send_marker(lsl_outlet, "NOGO_NORESPONSE")

    # win.flip()
    # trial 结束 marker
    # send_marker(lsl_outlet, f"TRIAL_END_{trial_index}")


def post_trial():
    # 空屏间隔
    win.flip()
    core.wait(0.5)


def init_exp(config: DictConfig | None = None):
    global n_blocks, n_trials_per_block, timing, stim_sequence

    n_blocks = config.n_blocks
    n_trials_per_block = config.n_trials_per_block
    timing = config.timing
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
    global win, clock, lsl_outlet, logger
    win = win_session if win_session else visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")
    clock = clock_session if clock_session else core.Clock()
    logger = logger_session if logger_session else setup_default_logger()

    lsl_outlet = lsl_outlet_session if lsl_outlet_session else init_lsl("GoNogoMarker")  # 初始化 LSL

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
