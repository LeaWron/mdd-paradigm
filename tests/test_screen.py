import pytest

from psycho.utils import PsychopyDisplaySelector, ScreenUtils


@pytest.mark.skip(reason="暂时不测试这个函数")
def test_all():
    screen_utils = ScreenUtils(show=True)

    print()
    print("=== 基本DPI感知设置测试 ===")
    screen_utils.set_dpi_awareness()

    print("\n=== 改进的显示器设备信息获取 ===")
    print(screen_utils.get_display_devices_dpi_aware())

    print("\n=== 改进的pyglet屏幕信息获取 ===")
    print(screen_utils.get_screen_pyglet_dpi_aware())

    print("\n=== 综合显示信息（包含物理尺寸）===")
    print(screen_utils.get_comprehensive_display_info())


@pytest.mark.skip(reason="暂时不测试这个函数")
def test_demo():
    """演示如何使用显示器选择器"""
    # 创建显示器选择器
    selector = PsychopyDisplaySelector()

    # 选择并预览显示器
    selected_screen = selector.select_and_preview()

    params = selector.get_selected_screen_window_params()["monitor"]
    if selected_screen is not None:
        print(f"选择了屏幕: {selected_screen}, 参数为: {params}")

        from psychopy import event, visual

        # 使用PsychoPy创建窗口
        win = visual.Window(
            size=params["pix_size"], fullscr=False, screen=selected_screen, color="grey"
        )

        # 显示选中的屏幕信息
        text = visual.TextStim(
            win=win, text=f"正在使用屏幕 {selected_screen + 1}", color="white"
        )
        text.draw()
        win.flip()

        # 等待按键关闭
        event.waitKeys()
        win.close()
    else:
        print("未选择屏幕")
