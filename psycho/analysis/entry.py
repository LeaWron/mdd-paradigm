from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from psycho.analysis.emotion_face import run_emotion_analysis
from psycho.analysis.prt import run_prt_analysis
from psycho.analysis.sret import run_sret_analysis
from psycho.analysis.utils import DataUtils, parse_date_input

# [ ] 根据需求进行分析
# 单受试, 分组 等等


def run_analysis(cfg: DictConfig, date: str = None, session_id: int = 0):
    data_utils = DataUtils(session_id=session_id, date=date)
    # 运行情感分析
    run_emotion_analysis(cfg, data_utils)
    # 运行PRT分析
    run_prt_analysis(cfg, data_utils)
    # 运行SRET分析
    run_sret_analysis(cfg, data_utils)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    date_input = input("请输入日期 (支持多种格式，缺省今天): \n")
    try:
        date = parse_date_input(date_input)
        print(f"解析后的日期: {date}")
    except ValueError as e:
        print(f"日期格式错误: {e}")
        print("使用今天的日期作为默认值")
        date = parse_date_input()

    session_id = input("请输入session ID (缺省则扫描日期下所有session): \n")

    if session_id:
        session_id = int(session_id)
        run_analysis(cfg, date, session_id)
    else:
        session_set = set()
        for s in (Path(cfg.output_dir) / date).iterdir():
            if s.is_file() and s.suffix == ".log":
                session_set.add(int(s.stem.split("-")[0]))

        for session_id in session_set:
            run_analysis(cfg, date, session_id)


if __name__ == "__main__":
    main()
