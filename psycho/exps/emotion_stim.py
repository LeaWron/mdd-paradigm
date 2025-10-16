import random
from collections import deque
from pathlib import Path

from omegaconf import DictConfig
from psychopy import core, event, sound, visual
from pylsl import StreamOutlet

from psycho.exps.resting import notification
from psycho.utils import adapt_image_stim_size, init_lsl, parse_stim_path, send_marker

# === 参数设置 ===
n_blocks = 1
n_trials_per_block = 1
continue_keys = ["space"]

timing = {
    "fixation": 2,
    "stim": 4,
    "recovery": 180,
}
stim_folder = parse_stim_path("emotion-stim")  # 刺激文件夹
stim_items = list(stim_folder.glob("*"))
stim_threshold = min(4, len(stim_items) - 1)  # 每个刺激出现的间隔的最小次数, 即距离它上次出现至少隔着几个不同的刺激

mathematic_trials_per_block = 1

# === 全局变量 ===
win = None
clock = None
lsl_outlet = None
port = None
block_index = 0
trial_index = 0

stim_set = set()
stim_deque: deque[Path] = deque()


# === 实验部分 ===
def pre_block():
    text = visual.TextStim(
        win,
        text=f"开始 Block {block_index + 1}，请按空格继续",
        color="white",
    )
    text.draw()
    win.flip()
    event.waitKeys(keyList=continue_keys)


def block():
    global trial_index
    for local_trial_index in range(n_trials_per_block):
        trial_index = local_trial_index
        pre_trial()
        trial()
        post_trial()


def post_block():
    """改为简单算术题任务"""

    def raise_question():
        # 生成随机算术题
        def get_question():
            # 确保算术题的结果在 0-100 之间
            while True:
                ops = ["+", "-", "*", "/"]
                op = random.choice(ops)
                a, b = random.randint(1, 99), random.randint(1, 99)
                if op == "/" and a % b != 0:
                    continue
                elif op == "*" and a * b >= 100:
                    a = random.randint(1, 99)
                    b = random.randint(1, max(100 // a, 1))

                question = f"{a} {op} {b} = "
                correct = eval(f"{a}{op}{b}")

                if correct < 0 or correct >= 100:
                    continue
                return question, correct

        question, correct = get_question()

        question_text = visual.TextStim(win, text=question, color="white", pos=(0, 0))
        answer_text = visual.TextStim(win, text="?", color="yellow", pos=(0.2, 0))
        prompt_text = visual.TextStim(win, text="请输入答案并按空格键确认", color="white", pos=(0, -0.3))

        response = ""
        while True:
            question_text.draw()
            answer_text.text = response
            answer_text.draw()
            prompt_text.draw()
            win.flip()

            keys = event.waitKeys()
            for key in keys:
                if key in ["space"]:
                    if response.strip() == "":
                        continue
                    try:
                        ans = int(response)
                    except ValueError:
                        ans = None
                    feedback = "正确！" if ans == correct else f"错误！正确答案是 {correct}"
                    fb = visual.TextStim(win, text=feedback, color="white")
                    fb.draw()
                    win.flip()
                    core.wait(1.5)
                    return
                elif key in ["backspace"]:
                    response = response[:-1]
                elif key.isdigit():
                    response += key

    # 做算术题
    # for _ in range(mathematic_trials_per_block):
    #     raise_question()

    # 恢复
    recovery_text = visual.TextStim(win, text="该区块完成, 请闭眼静坐\n休息 3 分钟, 直到再次听到如下提示", color="white", pos=(0, 0), height=0.05, wrapWidth=2)
    recovery_text.draw()

    sount_prompt = sound.Sound(notification, secs=1)
    win.flip()
    sount_prompt.play()
    send_marker(lsl_outlet, "RECOVERY_START")
    core.wait(timing["recovery"])
    sount_prompt.play()
    send_marker(lsl_outlet, "RECOVERY_END")
    core.wait(1)

    arousal_prompt, arousal_stim = rating_slider("请自评 arousal", up=True)
    valence_prompt, valence_stim = rating_slider("请自评 valence", up=False)
    confirm_text = visual.TextStim(win, text="按空格确认", color="white", pos=(0, -0.6))

    while True:
        arousal_prompt.draw()
        arousal_stim.draw()

        valence_prompt.draw()
        valence_stim.draw()
        confirm_text.draw()
        win.flip()
        keys = event.getKeys(continue_keys)
        if "space" in keys:
            arousal = arousal_stim.getRating()
            valence = valence_stim.getRating()
            break
    core.wait(0.2)


def pre_trial():
    # fixation
    fixation = visual.TextStim(win, text="+", color="white", height=0.4, wrapWidth=2)
    fixation.draw()
    win.flip()
    core.wait(timing["fixation"])


def trial():
    # 呈现图片
    def random_stim():
        while True:
            stim_item = random.choice(stim_items)
            if stim_item not in stim_set:
                stim_set.add(stim_item)
                stim_deque.append(stim_item)
                if len(stim_set) == stim_threshold:
                    stim_set.remove(stim_deque.popleft())
                break

        stim_height, aspect_ratio = adapt_image_stim_size(stim_item)
        stim = visual.ImageStim(
            win,
            image=stim_item,
            size=(stim_height * aspect_ratio, stim_height),
        )
        return stim

    stim = random_stim()
    stim.draw()
    win.flip()
    send_marker(lsl_outlet, "TRIAL_START")
    core.wait(timing["stim"])


def post_trial():
    # rating + resting
    # Valence rating
    valence_question, valence_slider = rating_slider("请评价愉快程度 (1=非常消极, 9=非常积极)", up=True)

    # Arousal rating
    arousal_question, arousal_slider = rating_slider("请评价唤醒度 (1=非常平静, 9=非常激动)", up=False)
    confirm_text = visual.TextStim(win, text="按空格确认", color="white", pos=(0, -0.6))
    while True:
        valence_question.draw()
        valence_slider.draw()
        arousal_question.draw()
        arousal_slider.draw()
        confirm_text.draw()
        win.flip()

        keys = event.getKeys()
        if "space" in keys:
            # TODO: 是否要用按钮来确认
            valence = valence_slider.getRating()
            arousal = arousal_slider.getRating()
            print(f"Valence: {valence}, Arousal: {arousal}")
            break
    send_marker(lsl_outlet, "TRIAL_END")
    core.wait(0.2)


# ========== rating 界面 ==========
def rating_slider(prompt, labels=None, up=True):
    pos_sign = 1 if up else -1
    question = visual.TextStim(win, text=prompt, color="white", pos=(0, pos_sign * 0.4), wrapWidth=2)
    labels = labels or [1, 2, 3, 4, 5, 6, 7, 8, 9]
    slider = visual.Slider(
        win,
        ticks=labels,
        labels=labels,
        granularity=0,  # 连续可拖动
        startValue=labels[len(labels) // 2],
        size=(0.9, 0.05),
        style=["rating"],
        color="white",
        pos=(0, pos_sign * 0.2),
    )
    # slider.setValue(labels[len(labels) // 2])
    slider.setValue(slider.startValue)
    return question, slider


def baseline_assess():
    baseline_text = visual.TextStim(win, text="请闭眼静坐 3 分钟, 直到再次听到如下提示", color="white", pos=(0, 0), wrapWidth=2)
    baseline_text.draw()

    sount_prompt = sound.Sound(notification, secs=1)
    win.flip()
    send_marker(lsl_outlet, "BASELINE_START")
    sount_prompt.play()
    core.wait(timing["recovery"])
    sount_prompt.play()

    send_marker(lsl_outlet, "BASELINE_END")
    core.wait(1)

    arousal_prompt, arousal_stim = rating_slider("请自评 arousal", up=True)
    valence_prompt, valence_stim = rating_slider("请自评 valence", up=False)
    confirm_text = visual.TextStim(win, text="按空格确认", color="white", pos=(0, -0.6))

    while True:
        arousal_prompt.draw()
        arousal_stim.draw()

        valence_prompt.draw()
        valence_stim.draw()
        confirm_text.draw()
        win.flip()
        keys = event.getKeys(continue_keys)
        if "space" in keys:
            arousal = arousal_stim.getRating()
            valence = valence_stim.getRating()
            break
    core.wait(0.2)


def init_exp(config: DictConfig | None = None):
    global n_blocks, n_trials_per_block, timing, stim_folder, stim_deque, stim_items, stim_set, stim_threshold, mathematic_trials_per_block

    n_blocks = config.n_blocks
    n_trials_per_block = config.n_trials_per_block
    timing = config.timing
    mathematic_trials_per_block = config.mathematic_trials_per_block
    stim_folder = parse_stim_path(config.stim_folder)
    stim_items = list(stim_folder.glob("*"))
    stim_deque = deque(maxlen=mathematic_trials_per_block)
    stim_set = set()
    stim_threshold = mathematic_trials_per_block


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
):
    global win, clock, lsl_outlet, block_index
    win = win_session if win_session else visual.Window(pos=(0, 0), fullscr=True, color="grey", units="norm")

    clock = clock_session if clock_session else core.Clock()

    lsl_outlet = lsl_outlet_session if lsl_outlet_session else init_lsl("EmotionStimMarker")  # 初始化 LSL

    baseline_assess()

    if config is not None and "pre" in config:
        run_exp(config.pre)

    send_marker(lsl_outlet, "EXPERIMENT_START")
    run_exp(config.full if config is not None else None)
    send_marker(lsl_outlet, "EXPERIMENT_END")


def main():
    entry()


if __name__ == "__main__":
    main()
