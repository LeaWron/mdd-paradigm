import multiprocessing
from pathlib import Path

from psychopy import session

from psycho.lsl_recv import main as lsl_recv_main


def start_session():
    """启动实验会话"""
    sess = session.Session(Path("./").absolute())
    exp_root = Path(__file__).parent / "exps"
    sess.addExperiment(exp_root / "gng.py", key="go-nogo")
    sess.addExperiment(exp_root / "nback.py", key="nback")

    # start_listener()
    # 会阻塞，直到所有实验结束
    sess.start()


def start_listener():
    """启动 LSL 监听器"""
    listener_process = multiprocessing.Process(target=lsl_recv_main)
    listener_process.start()


def main():
    start_session()


if __name__ == "__main__":
    main()
