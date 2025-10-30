from omegaconf import DictConfig
from psychopy import core, event, visual

from psycho.session import Experiment

# TODO: 是否需要发送 marker
# === 参数设置
timing = {
    "fixation": 0.5,
    "iti": 0.5,
}

# === 全局变量 ===
win = None
clock = None

questionnaires: dict = {}
questionnaire: dict = None

item: dict = None
score = 0


def pre_block():
    # 显示一个对话框，询问用户是否要继续
    pass


def block():
    global item
    # questionnaire["items"]: list[dict]
    for local_item in questionnaire["items"]:
        item = local_item

        pre_trial()
        trial()
        post_trial()


def post_block():
    # 显示一个对话框，询问用户是否要继续
    pass


def pre_trial():
    fixation = visual.TextStim(win, text="+", height=0.4)
    fixation.draw()
    win.flip()
    core.wait(timing["fixation"])


def trial():
    global score
    item_text = visual.TextStim(
        win, text=item["text"], height=0.12, wrapWidth=2, pos=(0, 0.4)
    )
    label_info = item["label"]
    labels = [label_info[i]["tip"] for i in range(len(label_info))]
    item_label = visual.Slider(
        win,
        ticks=list(range(len(labels))),
        labels=labels,
        labelWrapWidth=None,
        font="SimSun",
        granularity=1,
        size=(1.2, 0.12),
        style=["radio"],
    )

    base_color = [-0.2, -0.2, -0.2]
    hover_color = [0.1, 0.1, 0.1]
    disabled_color = [-0.5, -0.5, -0.5]

    button_box = visual.Rect(
        win,
        width=0.4,
        height=0.3,
        pos=(0, -0.5),
        fillColor=disabled_color,
        lineColor="white",
    )

    button_text = visual.TextStim(win, text="确认", height=0.1, pos=(0, -0.5))

    mouse = event.Mouse(win=win)

    while True:
        item_text.draw()
        item_label.draw()

        if item_label.markerPos is None:
            button_box.fillColor = disabled_color
        else:
            if button_box.contains(mouse):
                button_box.fillColor = hover_color
            else:
                button_box.fillColor = base_color

        button_box.draw()
        button_text.draw()
        win.flip()

        if item_label.rating is not None and mouse.isPressedIn(button_box):
            idx = item_label.getRating()
            value = label_info[idx]["value"]
            print(label_info[idx]["tip"], value)
            score += value
            break


def post_trial():
    win.flip()
    core.wait(timing["iti"])


def save_result():
    # questionnaire_name = questionnaire["name"]
    pass


def init_exp(cfg: DictConfig):
    global questionnaires, timing
    questionnaires = cfg.questionnaires
    timing = cfg.timing


def run_exp(cfg: DictConfig):
    global questionnaire

    if cfg is not None:
        init_exp(cfg)

    for question_block in list(questionnaires.values()):
        questionnaire = question_block

        pre_block()
        block()
        post_block()

        save_result()


def entry(exp: Experiment | None = None):
    global clock, win
    win = exp.win or visual.Window(fullscr=True, units="norm")
    clock = exp.clock or core.Clock()

    run_exp(exp.config)


def main():
    entry()


if __name__ == "__main__":
    # 显示一个对话框，询问用户是否要继续
    main()
