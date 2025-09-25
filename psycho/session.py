import multiprocessing
from pathlib import Path

from lsl_recv import main as lsl_recv_main
from psychopy import session


def start_session():
    """启动实验会话"""
    sess = session.Session(Path("./").absolute())
    a = sess.addExperiment("gng.py", key="go-nogo")
    print(a)

    # start_listener()
    sess.start()


def start_listener():
    """启动 LSL 监听器"""
    listener_process = multiprocessing.Process(target=lsl_recv_main)
    listener_process.start()


def main():
    start_session()


if __name__ == "__main__":
    main()
