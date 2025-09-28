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
