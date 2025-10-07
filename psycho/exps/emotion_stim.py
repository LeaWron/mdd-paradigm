import random
from collections import deque
from pathlib import Path

from PIL import Image
from psychopy import core, event, visual

from psycho.utils import adapt_image_stim_size, init_lsl, parse_stim_path, send_marker

# === 参数设置 ===
n_blocks = 1
n_trials_per_block = 1
continue_keys = ["space"]

fixation_duration = 2

stim_folder = parse_stim_path("emotion-stim")  # 刺激文件夹
stim_items = list(stim_folder.glob("*"))
stim_threshold = min(4, len(stim_items) - 1)  # 每个刺激出现的间隔的最小次数, 即距离它上次出现至少隔着几个不同的刺激
stim_duration = 5

mathematic_trials_per_block = 10

# === 全局变量 ===
win = None
clock = None
lsl_outlet = None
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

    for _ in range(mathematic_trials_per_block):
        raise_question()


def pre_trial():
    # fixation
    fixation = visual.TextStim(win, text="+", color="white", height=0.4, wrapWidth=2)
    fixation.draw()
    win.flip()
    core.wait(fixation_duration)


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
    core.wait(stim_duration)


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
        startValue=labels[len(labels) // 2],  # TODO: 不移动值依然为 None
        size=(0.9, 0.05),
        style=["rating"],
        color="white",
        pos=(0, pos_sign * 0.2),
    )
    return question, slider


def entry(
    win_session: visual.Window | None = None,
    clock_session: core.Clock | None = None,
):
    global win, clock, lsl_outlet, block_index
    win = visual.Window(fullscr=True, color="grey", units="norm") if win_session is None else win_session
    clock = core.Clock() if clock_session is None else clock_session
    lsl_outlet = init_lsl("EmotionStim")

    for local_block_index in range(n_blocks):
        block_index = local_block_index
        pre_block()
        block()
        post_block()

    send_marker(
        lsl_outlet,
        "EXPERIMENT_END",
    )


def main():
    entry()


if __name__ == "__main__":
    main()
