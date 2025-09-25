from psychopy import core, event, gui, prefs, visual

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
