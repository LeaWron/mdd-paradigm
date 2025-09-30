import random

from psychopy import core, event, visual

from psycho.utils import init_lsl, parse_stim_path, send_marker

# === 实验参数 ===
blocks_info = [
    {
        "n_trials": 0,
        "prompt": """接下来的任务中，将要求你对一组呈现的词语或图片进行分类。分类要尽可能地快，但同时又尽可能少犯错。
下面列出了类别标签以及属于那些类别的项目。
按空格键继续
""",
        "tips": "占位项, 让下标从 1 开始，同时也是实验开始的说明",
    },
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

key_blocks = [4, 7]  # 重要区块的索引

resp_keys = ["f", "j"]  # 反应键, f 对应左边, j 对应右边

continue_key = ["space"]  # 继续键

max_wait_respond = 10.0  # 最大等待时间

# === 全局设置 ===
win = None
clock = None
lsl_outlet = None

correct_count = 0
block_index = 0  # 当前区块索引

# === 刺激设置 ===
stim_kind = ["自我", "他人", "生命", "死亡"]
stim_texts = {
    "自我": [
        "我",
        "自己",
        "我的",
        "俺",
        "咱",
        "本身",
        "本人",
        "自我",
        "自个儿",
        "吾",
        "余",
        "我辈",
        "吾身",
        "吾辈",
        "在下",
        "本我",
        "自家",
        "自个",
        "本人自身",
        "我自己",
    ],
    "他人": [
        "他",
        "她",
        "它",
        "他们",
        "她们",
        "它们",
        "其",
        "彼",
        "彼人",
        "他者",
        "他人",
        "他辈",
        "他身",
        "她身",
        "其人",
        "某人",
        "某他",
        "旁人",
    ],
    "生命": [
        "生命",
        "生存",
        "生活",
        "活力",
        "存在",
        "生机",
        "成长",
        "生命力",
        "呼吸",
        "心跳",
        "意识",
        "灵魂",
        "生命线",
        "生长",
        "繁衍",
        "生育",
        "活着",
        "生命体",
        "生命现象",
        "生灵",
    ],
    "死亡": [
        "死亡",
        "逝去",
        "去世",
        "故去",
        "亡故",
        "离世",
        "谢世",
        "长逝",
        "亡",
        "逝者",
        "终结",
        "消亡",
        "死去",
        "辞世",
        "殒命",
        "过世",
        "死",
        "死去的人",
        "命终",
        "夭折",
    ],
}


def pre_block(block_index: int, test_mode: bool = False):
    """block 开始前"""
    if block_index == 0:
        show_prompt()
    else:
        visual.TextStim(
            win=win,
            text=blocks_info[block_index]["prompt"],
            height=0.05,
            wrapWidth=2,
            alignText="center",
        ).draw()

    win.flip()
    event.waitKeys(keyList=continue_key)
    send_marker(lsl_outlet, f"BLOCK_START_{block_index}", clock.getTime())


def block(block_index: int):
    """block 运行中"""
    n_trials = blocks_info[block_index]["n_trials"]
    for trial_index in range(n_trials):
        pre_trial(trial_index)
        trial(trial_index)
        post_trial(trial_index)

    send_marker(lsl_outlet, f"BLOCK_END_{block_index}", clock.getTime())


def post_block(block_index: int, test_mode: bool = False):
    """block 结束后"""
    global correct_count
    win.flip()
    if block_index == 0:
        return
    correct_rate = correct_count / blocks_info[block_index]["n_trials"]
    correct_count = 0
    visual.TextStim(
        win=win,
        text=f"在这个任务中你的正确率为: {correct_rate * 100:.2f}%\n按空格键继续",
        height=0.1,
        wrapWidth=2,
    ).draw()
    win.flip()
    event.waitKeys(keyList=continue_key)


def pre_trial(trial_index: int):
    """trial 开始前"""
    fixation = visual.TextStim(win=win, text="+", height=0.4)
    fixation.draw()
    win.flip()
    core.wait(0.5)


def trial(trial_index: int):
    """trial 运行中"""
    global correct_count

    def show_stim():
        """显示刺激"""
        left_kinds: list[str] = blocks_info[block_index]["left_kinds"]
        right_kinds: list[str] = blocks_info[block_index]["right_kinds"]
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
        l_or_r = random.choice(["left", "right"])
        if l_or_r == "left":
            stim_range = []
            for kind in left_kinds:
                stim_range.extend(stim_texts[kind])
            stim_text = random.choice(stim_range)
        else:
            stim_range = []
            for kind in right_kinds:
                stim_range.extend(stim_texts[kind])
            stim_text = random.choice(stim_range)
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
    resp = event.waitKeys(maxWait=max_wait_respond, keyList=resp_keys, timeStamped=True)
    correct = False
    if resp is None:
        send_marker(lsl_outlet, f"TRIAL_{trial_index}_NO_RESPONSE", clock.getTime())
    elif resp[0][0] == stim_correct_resp:
        correct = True

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

    send_marker(lsl_outlet, f"TRIAL_{trial_index}_{resp}", clock.getTime())
    pass


def post_trial(trial_index: int):
    """trial 结束后"""
    win.flip()
    pass


def entry(win_session: visual.Window | None = None, clock_session: core.Clock | None = None, test_mode: bool = False):
    """实验入口"""
    global win, clock, lsl_outlet, block_index
    win = win_session if win_session is not None else visual.Window(fullscr=True, units="norm")
    clock = clock_session if clock_session is not None else core.Clock()
    lsl_outlet = init_lsl("Diat")

    n_blocks = len(blocks_info)
    for bi in range(n_blocks):
        block_index = bi
        pre_block(block_index, test_mode)
        block(block_index)
        post_block(block_index, test_mode)

    send_marker(lsl_outlet, "EXPERIMENT_END", clock.getTime())


def main():
    """实验主函数"""
    entry(test_mode=True)


if __name__ == "__main__":
    main()


def show_prompt():
    """显示提示"""
    visual.TextStim(
        win=win,
        text=blocks_info[0]["prompt"],
        height=0.05,
        wrapWidth=1.8,
    ).draw()

    visual.ImageStim(
        win=win,
        image=parse_stim_path("image_item_table.png"),
        pos=(0, -0.6),
    ).draw()
