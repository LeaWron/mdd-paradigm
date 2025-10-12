import pytest
from psychopy import core, event, gui, prefs, visual


@pytest.mark.skip(reason="暂时不测试这个函数")
def test_gui():
    myDlg = gui.Dlg(title="Go-nogo 任务")
    myDlg.addText("请输入被试信息")
    myDlg.addField(key="name", label="被试姓名:")
    myDlg.addField("age", 18, label="被试年龄:")
    myDlg.addText("请输入实验信息")
    myDlg.addField("刺激方向:", 45)
    myDlg.addField("组别:", choices=["Test", "Control"])
    ok_data = myDlg.show()  # show dialog and wait for OK or Cancel
    if myDlg.OK:  # or if ok_data is not None
        print(ok_data)
    else:
        print("user cancelled")


@pytest.mark.skip(reason="暂时不测试这个函数")
def test_clock():
    timer = core.Clock()
    print()
    print("1", timer.getTime())
    timer.addTime(5)
    print("2", timer.getTime())
    while 5.0001 > timer.getTime() > 0:
        print(timer.getTime())

    timer.reset()
    print("3", timer.getTime())


@pytest.mark.skip(reason="暂时不测试这个函数")
def test_text():
    # 创建窗口
    win = visual.Window([800, 600], color="black", units="norm")

    def create_textStim(win, text, height=0.1, pos=(0, 0), wrapWidth=None, autoDraw=False):
        """
        创建一个 TextStim，保证视觉居中，多行也居中，默认自动绘制。

        参数：
        - win: psychopy Window 对象
        - text: 要显示的文字（可含 \n 换行）
        - height: 字体高度
        - pos: 文字中心位置 (x, y)
        - wrapWidth: 自动换行宽度 (None 表示不限制)
        - autoDraw: 是否自动绘制
        """
        stim = visual.TextStim(
            win=win,
            text=text,
            height=height,
            pos=pos,
            wrapWidth=wrapWidth,
            autoDraw=autoDraw,
        )
        return stim

    # 示例 1: 单行文字
    stim1 = create_textStim(
        win,
        "Hello World But I need some more longer text so I can know whether it is right way I need",
        height=0.2,
    )
    stim1.draw()
    win.flip()
    event.waitKeys()

    # 示例 2: 多行文字 + wrapWidth
    multi_line_text = "第一行文字\n第二行文字\n第三行文字"
    stim2 = create_textStim(win, multi_line_text, height=0.1, wrapWidth=1.5)
    stim2.draw()
    win.flip()
    event.waitKeys()

    stim3 = create_textStim(
        win,
        "Hello World But I need some more longer text\nso I can know whether it is right way I need",
        height=0.2,
        wrapWidth=2,
    )
    stim3.draw()
    win.flip()
    event.waitKeys()
    # 显示示例
    win.close()


@pytest.mark.skip(reason="暂时不测试这个函数")
def test_switch_keyboard_layout():
    # 切换到英文输入法
    import ctypes

    # 加载 user32.dll
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    # HKL 对应输入法标识符：0x04090409 = en-US 键盘布局
    HKL_NEXT = 0x04090409

    # 切换输入法
    user32.ActivateKeyboardLayout(HKL_NEXT, 0)


@pytest.mark.skip(reason="暂时不测试这个函数")
def test_block_interaction():
    # orbitary_keys = [chr(i) for i in range(32, 127)] + ["return", "space"]
    orbitary_keys = None
    print(orbitary_keys)
    win = visual.Window([800, 600], color="black", units="norm")
    event.waitKeys(keyList=orbitary_keys)
    stim = visual.TextStim(win, text="+", height=0.2)
    stim.draw()
    win.flip()

    while True:
        keys = event.getKeys(keyList=orbitary_keys)
        if "space" in keys:
            print("你按下了以下键:", event.waitKeys(modifiers=True))
        if "return" in keys:
            break
    win.close()


@pytest.mark.skip(reason="暂时不测试这个函数")
def test_get_isi():
    import random
    import time

    while input() != "y":
        isi = random.uniform(500, 1000)
        print(isi)


@pytest.mark.skip(reason="暂时不测试这个函数")
def test_prev_frame():
    win = visual.Window(size=(600, 400), color="white")

    win.flip()
    text1 = visual.TextStim(win, text="我想保留的内容", pos=(0, 0.2), color="black")

    text2 = visual.TextStim(win, text="叠加的新内容", pos=(0, -0.2), color="blue")

    # 关键：在你要保留的那一帧就传入 clearBuffer=False
    text1.draw()
    win.flip(clearBuffer=False)  # <-- 把 False 放在这里，text1 的像素会留在 back buffer

    event.waitKeys()

    # 现在再画新的内容并 flip，会在保留的像素上叠加
    text2.draw()
    win.flip()

    event.waitKeys()
    win.close()


@pytest.mark.skip(reason="暂时不测试这个函数")
def test_get_devices():
    from psychopy import monitors, sound

    print(sound.getDevices())
    print(monitors.getAllMonitors())
    for monitor in monitors.getAllMonitors():
        monitor = monitors.Monitor(monitor)
        width = monitor.getWidth()
        distance = monitor.getDistance()
        size_pix = monitor.getSizePix()
        print(monitor, width, distance, size_pix)


def test_get_trial_sequence():
    from psycho.utils import generate_trial_sequence

    n_blocks = 2
    n_trials_per_block = 10
    max_seq_same = 2
    stim_list = ["A", "B", "C"]

    stim_sequences = generate_trial_sequence(n_blocks, n_trials_per_block, max_seq_same, stim_list)

    for block_index, seq in stim_sequences.items():
        print(f"Block {block_index}: {seq}")
