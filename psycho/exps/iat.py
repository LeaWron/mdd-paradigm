import numpy as np
from omegaconf import DictConfig
from psychopy import core, event, visual

from psycho.session import Experiment
from psycho.utils import (
    init_lsl,
    save_csv_data,
    send_marker,
    setup_default_logger,
    update_block,
    update_trial,
)

# TODO: 刺激词
# TODO: 无效被试实现
# TODO: 平均反应时间
# TODO: 结束后的有效性计算和D值计算
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
        "left_kinds": ["积极"],
        "right_kinds": ["消极"],
    },
    {
        "n_trials": 1,
        "prompt": """注意上方，之前各自呈现的四个类别标签现在一起出现。
记住，每个项目只属于一个类别。
例始，当类别标签花和好一起呈现在屏幕上方的左右两边时，属于花范畴的图片或词语需要被归纳到花这个类别，而不是好这个类别。

使用F键和J键将类别项目分类到左边和右边的四个类别中，纠正错误请按另一个键。

按下空格键开始""",
        "left_kinds": ["自我", "积极"],
        "right_kinds": ["他人", "消极"],
    },
    {
        "n_trials": 1,
        "prompt": """再次对同样的四个类别进行分类。记住尽可能快速且准确地作出反应。

使用F键和J键将类别项目分类到左边和右边的四个类别中，纠正错误请按另一个键。

按下空格键开始""",
        "left_kinds": ["自我", "积极"],
        "right_kinds": ["他人", "消极"],
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
        "left_kinds": ["他人", "积极"],
        "right_kinds": ["自我", "消极"],
    },
    {
        "n_trials": 1,
        "prompt": """再次对同样的四个类别进行分类。尽可能快速且准确地作出反应。

使用F键和J键将类别项目分类到左边和右边的四个类别中，纠正错误请按另一个键。

按下空格键开始""",
        "left_kinds": ["他人", "积极"],
        "right_kinds": ["自我", "消极"],
    },
]

key_blocks = [3, 6]  # 重要区块的索引

resp_keys = ["f", "j"]  # 反应键, f 对应左边, j 对应右边

continue_key = ["space"]  # 继续键

timing = {
    "max_wait_respond": 10.0,  # 最大等待时间
    "rest": 3.0,  # 休息时间
}

limits = {
    "lower": 0.3,  # 反应时下限，单位秒
    "upper": 10,  # 反应时上限，单位秒
}

effective_threshold = 0.9  # 有效反应阈值，单位百分比(%), 需要有效反应的次数占总次数的比例
warn_offset = 0.05  # 有效反应提示偏移，单位秒

err_offset = 0.6  # 错误反应时偏移，单位秒


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
    "积极": [
        "有价值",
        "成功",
        "高尚",
        "有意义",
        "能干",
        "骄傲",
        "可敬",
    ],
    "消极": [
        "失败",
        "无能",
        "软弱",
        "羞愧",
        "愧疚",
        "可耻",
    ],
}
stim_kinds = list(stims.keys())
stim_sequence = {
    0: ["我的", "他"],
    1: ["成功", "软弱"],
    2: ["自己", "能干", "他人", "可耻"],
    3: ["他人", "可敬", "自我", "失败"],
    4: ["自我", "他人"],
    5: ["自己", "能干", "他人", "可耻"],
    6: ["他人", "可敬", "自我", "失败"],
}

# === 数据保存 ===
data_to_save = {
    "exp_start_time": [],
    "exp_end_time": [],
    "block_index": [],
    "trial_index": [],
    "trial_start_time": [],
    "trial_end_time": [],
    "stim": [],
    "stim_kind": [],
    "choice": [],
    "rt": [],
    "correct_rate": [],
    "mean_rt": [],
    "effective": [],
    "d_score": [],
}

one_trial_data = {key: None for key in data_to_save.keys()}
one_block_data = {key: [] for key in data_to_save.keys()}


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
        one_trial_data["trial_index"] = trial_index
        # 记录 trial 开始时间
        one_trial_data["trial_start_time"] = clock.getTime()

        pre_trial()
        trial()
        post_trial()

        # 记录 trial 结束时间
        one_trial_data["trial_end_time"] = clock.getTime()

        update_trial(one_trial_data, one_block_data)


def post_block():
    global correct_count
    win.flip()
    correct_rate = correct_count / blocks_info[block_index]["n_trials"]
    correct_count = 0

    one_trial_data["correct_rate"] = correct_rate
    # 正确试次的平均反应时
    mean_rt = np.mean([rt for rt in one_block_data["rt"] if rt != float("inf")]).item()
    one_trial_data["mean_rt"] = mean_rt

    update_trial(one_trial_data, one_block_data)

    visual.TextStim(
        win=win,
        text=f"在这个任务中你的正确率为: {correct_rate * 100:.2f}%, 平均反应时为: {mean_rt * 1000:.2f}ms\n你有{int(timing['rest'])}秒休息时间\n你可以直接按空格键继续",
        height=0.1,
        wrapWidth=2,
    ).draw()

    # 添加有效反应提示
    if mean_rt < limits["lower"] + warn_offset:
        visual.TextStim(
            win=win,
            text="你的反应速度有点快，可能会影响任务的效果",
            height=0.1,
            wrapWidth=2,
        ).draw()
    elif mean_rt > limits["upper"] - warn_offset:
        visual.TextStim(
            win=win,
            text="你的反应速度有点慢，可能会影响任务的效果",
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

        one_trial_data["stim"] = stim_text
        one_trial_data["stim_kind"] = l_or_r

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
        on_set = clock.getTime()
        win.flip()

        # 可能还要再等一帧，确保所有刺激都被绘制
        left_stim.draw()
        right_stim.draw()
        stim.draw()
        if l_or_r == "left":
            return "f", on_set
        else:
            return "j", on_set

    stim_correct_resp, on_set = show_stim()
    # 记录刺激

    resp = event.waitKeys(maxWait=timing["max_wait_respond"], keyList=resp_keys, timeStamped=clock)
    correct = False
    # 反应时
    if resp is None:
        send_marker(lsl_outlet, "NORESPONSE")
        logger.info("No response")
    elif resp[0][0] == stim_correct_resp:
        correct = True
        send_marker(lsl_outlet, "CORRECT")

        one_trial_data["choice"] = resp[0][0]
        one_trial_data["rt"] = resp[0][1] - on_set
        logger.info(f"Correct: {stim_correct_resp}, rt: {one_trial_data['rt']:.3f}")
    else:
        one_trial_data["choice"] = resp[0][0]
        # 错误试次用 inf 标记
        one_trial_data["rt"] = float("inf")
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


def calculate_d_score(comp_rts: list[float], incomp_rts: list[float]) -> float:
    """计算 d-score"""

    # 错误反应用惩罚处理
    # 排除极端值
    def deal_rts(rts: list[float]) -> list[float]:
        """处理反应时"""
        d_rts = []
        err_indices = []
        for i, rt in enumerate(rts):
            if rt == float("inf"):
                err_indices.append(i)
            elif rt < limits["lower"] or rt > limits["upper"]:
                # 排除极端值
                continue
            d_rts.append(rt)
        # 错误反应时用惩罚处理
        mean_rt = np.mean(d_rts).item()
        for _ in err_indices:
            d_rts.append(mean_rt + err_offset)
        return d_rts

    comp_rts = deal_rts(comp_rts)
    incomp_rts = deal_rts(incomp_rts)

    mean_diff = np.mean(comp_rts) - np.mean(incomp_rts)
    std_comp = np.var(comp_rts)
    std_incomp = np.var(incomp_rts)
    std_pooled: float = np.sqrt((len(comp_rts) - 1) * std_comp + (len(incomp_rts) - 1) * std_incomp / (len(comp_rts) + len(incomp_rts) - 2))

    d_score = mean_diff / std_pooled
    return d_score.item()


def check_effective_and_calculate_d_score(data: dict[str, list]) -> float:
    """检查该受试数据是否有效, 并计算 d-score"""
    block_index_list = data["block_index"]

    rts = []
    for key_block in key_blocks:
        if key_block not in block_index_list:
            return False
        start = block_index_list.index(key_block)
        end = start + blocks_info[key_block]["n_trials"]  # [start, end)

        rt_list = data["rt"][start:end]
        rts.append(rt_list)

        cnt = 0
        for rt in rt_list:
            if rt >= limits["lower"]:
                cnt += 1

        if 1.0 * cnt / len(rt_list) < effective_threshold:
            return float("inf")

    d_score = calculate_d_score(comp_rts=rts[0], incomp_rts=rts[1])
    return d_score


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
    global blocks_info, key_blocks, start_prompt, timing, stims, stim_sequence, limits, effective_threshold, warn_offset, err_offset

    blocks_info = config.blocks_info
    key_blocks = config.key_blocks
    start_prompt = config.start_prompt
    timing = config.timing

    # pre 和 full 的配置不同
    if "stims" in config:
        stims = config.stims
    if "stim_sequence" in config:
        stim_sequence = config.stim_sequence
    if "limits" in config:
        limits = config.limits
    if "effective_threshold" in config:
        effective_threshold = config.effective_threshold
    if "warn_offset" in config:
        warn_offset = config.warn_offset
    else:
        warn_offset = 0.0
    if "err_offset" in config:
        err_offset = config.err_offset


def run_exp(cfg: DictConfig | None):
    global block_index

    if cfg is not None:
        init_exp(cfg)

    show_prompt()

    for local_block_index in range(len(blocks_info)):
        block_index = local_block_index
        one_trial_data["block_index"] = block_index

        pre_block()
        block()
        post_block()

        update_block(one_block_data, data_to_save)

    # 记录 trial 数据
    one_trial_data["block_index"] = -1
    d_score = check_effective_and_calculate_d_score(data_to_save)
    # 判断有效性
    if d_score == float("inf"):
        one_trial_data["effective"] = False
    else:
        one_trial_data["effective"] = True
        one_trial_data["d_score"] = d_score

    update_trial(one_trial_data, one_block_data)
    update_block(one_block_data, data_to_save)


def entry(exp: Experiment | None = None):
    """实验入口"""
    global win, clock, lsl_outlet, logger
    win = exp.win or visual.Window(fullscr=True, units="norm")
    clock = exp.clock or core.Clock()
    logger = exp.logger or setup_default_logger()

    lsl_outlet = exp.lsl_outlet or init_lsl("D-IATMarker")

    one_trial_data["exp_start_time"] = clock.getTime()
    send_marker(lsl_outlet, "EXPERIMENT_START")
    logger.info("实验开始")

    run_exp(exp.config.full if exp.config is not None else None)

    # 记录实验结束时间
    one_trial_data["exp_end_time"] = clock.getTime()

    send_marker(lsl_outlet, "EXPERIMENT_END")
    logger.info("实验结束")

    if exp is not None:
        logger.info("保存数据")
        update_trial(one_trial_data, one_block_data)
        update_block(one_block_data, data_to_save)

        save_csv_data(data_to_save, exp.session_info["save_path"] + "-iat")


def main():
    entry()


if __name__ == "__main__":
    main()
