import logging

from omegaconf import DictConfig
from psychopy import core, event, visual
from pylsl import StreamOutlet

from psycho.utils import init_lsl, send_marker, setup_default_logger

# === 实验参数 ===
blocks_info = [
    {
        "n_trials": 1,
        "prompt": """请以舒适的姿势将两根任意手指放在键盘F键和J键上。
属于不同类别的词语或图像将一个一个地呈现在屏幕中心，这些类别标签将始终显示在屏幕上方两侧。
当呈现的项目属于左边的类别时，请按F键；当呈现的项目属于右边类别时，请按J键。每个项目只属于一个类别。
如果按键错误，将出现X，需要按另一个键修正并继续进行。
这是一个计时分类任务。需要你尽可能快且准确地进行反应。这个任务需要大约十五分钟时间完成。

按下空格键开始""",
        "left_kinds": ["自我"],
        "right_kinds": ["他人"],
    },
    {
        "n_trials": 1,
        "prompt": """注意上方，类别标签已改变。需要分类的项目也改变。但是规则未改变。
当呈现的项目属于左边的类别时，请按F键；当呈现的项目属于右边类别时，请按J键。每个项目只属于一个类别。
如果按键错误，将出现X，需要按另一个键纠正并继续进行。尽可能快速作出反应。

按下空格键开始""",
        "left_kinds": ["生命"],
        "right_kinds": ["死亡"],
    },
    {
        "n_trials": 1,
        "prompt": """注意上方，之前各自呈现的四个类别标签现在一起出现。
记住，每个项目只属于一个类别。
例始，当类别标签花和好一起呈现在屏幕上方的左右两边时，属于花范畴的图片或词语需要被归纳到花这个类别，而不是好这个类别。

使用F键和J键将类别项目分类到左边和右边的四个类别中，纠正错误请按另一个键。

按下空格键开始""",
        "left_kinds": ["自我", "生命"],
        "right_kinds": ["他人", "死亡"],
    },
    {
        "n_trials": 1,
        "prompt": """再次对同样的四个类别进行分类。记住尽可能快速且准确地作出反应。

使用F键和J键将类别项目分类到左边和右边的四个类别中，纠正错误请按另一个键。

按下空格键开始""",
        "left_kinds": ["自我", "生命"],
        "right_kinds": ["他人", "死亡"],
    },
    {
        "n_trials": 1,
        "prompt": """注意上方，只有两个类别标签，但互换了位置。
此前在左边的类别标签现在在右边，而此前在右边的类别标签现在在左边。请练习新的安排。

按F键和J键进行项目的左右分类，按另一个键可以纠正错误。

按下空格键开始""",
        "left_kinds": ["他人"],
        "right_kinds": ["自我"],
    },
    {
        "n_trials": 1,
        "prompt": """注意上方，四个类别标签以新的组合方式出现。记住，每个项目只属于一个类别。

使用F键和J键将类别项目分类到左边和右边的四个类别中，纠正错误请按另一个键。

按下空格键开始""",
        "left_kinds": ["他人", "生命"],
        "right_kinds": ["自我", "死亡"],
    },
    {
        "n_trials": 1,
        "prompt": """再次对同样的四个类别进行分类。尽可能快速且准确地作出反应。

使用F键和J键将类别项目分类到左边和右边的四个类别中，纠正错误请按另一个键。

按下空格键开始""",
        "left_kinds": ["他人", "生命"],
        "right_kinds": ["自我", "死亡"],
    },
]

key_blocks = [3, 6]  # 重要区块的索引

resp_keys = ["f", "j"]  # 反应键, f 对应左边, j 对应右边

continue_key = ["space"]  # 继续键

timing = {
    "max_wait_respond": 10.0,  # 最大等待时间
    "rest": 3.0,  # 休息时间
}

# === 全局设置 ===
win = None
clock = None
lsl_outlet = None
logger = None

correct_count = 0
block_index = 0  # 当前区块索引
trial_index = 0

start_prompt = """接下来的任务中，将要求你对一组呈现的词语或图片进行分类。分类要尽可能地快，但同时又尽可能少犯错。
下面列出了类别标签以及属于那些类别的项目。

按空格键继续
"""


# === 刺激设置 ===
stims = {
    "自我": ["我", "自己", "我的", "本人", "自我", "我辈"],
    "他人": ["他", "她", "它", "他们", "她们", "它们", "他人", "某人", "旁人"],
    "生命": ["生命", "生存", "生活", "活力", "存在", "生机", "成长", "呼吸", "心跳", "生长", "繁衍", "生育", "活着"],
    "死亡": ["死亡", "逝去", "去世", "故去", "亡故", "离世", "逝者", "终结", "消亡", "死去", "辞世", "殒命", "过世", "死", "夭折"],
}
stim_kinds = list(stims.keys())
stim_sequence = {
    0: ["自我", "他人"],
    1: ["生命", "死亡"],
    2: ["自我", "生命", "他人", "死亡"],
    3: ["他人", "生命", "自我", "死亡"],
    4: ["生命", "死亡", "自我", "他人"],
    5: ["生命", "死亡", "他人", "自我"],
    6: ["他人", "死亡", "自我", "生命"],
}


def pre_block():
    visual.TextStim(
        win=win,
        text=blocks_info[block_index]["prompt"],
        height=0.05,
        wrapWidth=2,
        alignText="center",
    ).draw()

    win.flip()
    event.waitKeys(keyList=continue_key)


def block():
    global trial_index
    n_trials = blocks_info[block_index]["n_trials"]
    for local_trial_index in range(n_trials):
        trial_index = local_trial_index
        pre_trial()
        trial()
        post_trial()


def post_block():
    global correct_count
    win.flip()
    correct_rate = correct_count / blocks_info[block_index]["n_trials"]
    correct_count = 0
    visual.TextStim(
        win=win,
        text=f"在这个任务中你的正确率为: {correct_rate * 100:.2f}%\n你有{int(timing['rest'])}秒休息时间\n你可以直接按空格键继续",
        height=0.1,
        wrapWidth=2,
    ).draw()
    win.flip()
    event.waitKeys(keyList=continue_key)


def pre_trial():
    """trial 开始前"""
    fixation = visual.TextStim(win=win, text="+", height=0.4)
    fixation.draw()
    win.flip()
    core.wait(0.5)


def trial():
    """trial 运行中"""
    global correct_count

    def show_stim():
        """显示刺激"""
        left_kinds: list[str] = blocks_info[block_index]["left_kinds"]
        right_kinds: list[str] = blocks_info[block_index]["right_kinds"]

        left_stims = [stim for kind in left_kinds for stim in stims[kind]]
        color = "#00ff00"
        color_space = "hex"
        # 左
        left_stim = visual.TextStim(
            win=win,
            text="\n\n".join(left_kinds),
            pos=(-0.4, 0.4),
            height=0.05,
            color=color,
            colorSpace=color_space,
        )
        # 右
        right_stim = visual.TextStim(
            win=win,
            text="\n\n".join(right_kinds),
            pos=(0.4, 0.4),
            height=0.05,
            color=color,
            colorSpace=color_space,
        )

        # 中心刺激
        stim_text = stim_sequence[block_index][trial_index]
        l_or_r = "left" if stim_text in left_stims else "right"

        stim = visual.TextStim(
            win=win,
            text=stim_text,
            height=0.1,
            wrapWidth=2,
        )

        # 绘制所有刺激
        left_stim.draw()
        right_stim.draw()
        stim.draw()

        send_marker(lsl_outlet, "TRIAL_START")
        win.flip()

        # 可能还要再等一帧，确保所有刺激都被绘制
        left_stim.draw()
        right_stim.draw()
        stim.draw()
        if l_or_r == "left":
            return "f"
        else:
            return "j"

    stim_correct_resp = show_stim()
    resp = event.waitKeys(maxWait=timing["max_wait_respond"], keyList=resp_keys, timeStamped=True)
    correct = False
    # 反应时
    if resp is None:
        send_marker(lsl_outlet, "NORESPONSE")
        logger.info("No response")
    elif resp[0][0] == stim_correct_resp:
        correct = True
        send_marker(lsl_outlet, "CORRECT")
        logger.info(f"Correct: {stim_correct_resp}")
    else:
        send_marker(lsl_outlet, "INCORRECT")
        logger.info(f"Incorrect: {stim_correct_resp}")

    if correct:
        correct_count += 1
        win.flip()
    else:
        visual.TextStim(
            win=win,
            text="x",
            color="red",
            height=0.2,
            pos=(0, -0.2),
        ).draw()
        win.flip()
        event.waitKeys(keyList=[stim_correct_resp])


def post_trial():
    """trial 结束后"""
    win.flip()
    core.wait(0.1)


def show_prompt():
    """显示提示"""
    visual.TextStim(
        win=win,
        text=start_prompt,
        height=0.05,
        wrapWidth=1.8,
    ).draw()

    for i, (key, value) in enumerate(stims.items()):
        visual.TextStim(
            win=win,
            text=f"{key}: {', '.join(value)}",
            height=0.05,
            pos=(0, -0.2 - 0.07 * (i + 1)),
            wrapWidth=1.8,
            alignText="left",
        ).draw()
    win.flip()
    event.waitKeys(keyList=continue_key)


def init_exp(config: DictConfig | None):
    global blocks_info, key_blocks, start_prompt, timing, stims, stim_sequence

    blocks_info = config.blocks_info
    key_blocks = config.key_blocks
    start_prompt = config.start_prompt
    timing = config.timing
    stims = config.stims
    if "stim_sequence" in config:
        stim_sequence = config.stim_sequence


def run_exp(cfg: DictConfig | None):
    global block_index

    if cfg is not None:
        init_exp(cfg)

    show_prompt()

    for local_block_index in range(len(blocks_info)):
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
    global win, clock, lsl_outlet, logger
    win = win_session if win_session is not None else visual.Window(fullscr=True, units="norm")
    clock = clock_session if clock_session is not None else core.Clock()
    logger = logger_session if logger_session is not None else setup_default_logger()

    lsl_outlet = lsl_outlet_session if lsl_outlet_session else init_lsl("D-IATMarker")

    send_marker(lsl_outlet, "EXPERIMENT_START")
    logger.info("实验开始")
    run_exp(config.full if config is not None else None)
    send_marker(lsl_outlet, "EXPERIMENT_END")
    logger.info("实验结束")


def main():
    entry()


if __name__ == "__main__":
    main()
