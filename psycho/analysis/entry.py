import hydra
from omegaconf import DictConfig, OmegaConf

from psycho.analysis.emotion_face import run_emotion_analysis
from psycho.analysis.prt import run_prt_analysis
from psycho.analysis.sret import run_sret_analysis
from psycho.analysis.utils import DataUtils

# [ ] 根据需求进行分析
# 单受试, 分组 等等


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    date = input("请输入日期 (YYYY-MM-DD, 缺省当前日期): \n")
    date = date if date else None

    session_id = int(input("请输入session ID: \n"))
    if date:
        data_utils = DataUtils(session_id=session_id, date=date)
    else:
        data_utils = DataUtils(session_id=session_id)

    # 运行情感分析
    run_emotion_analysis(cfg, data_utils)
    # 运行PRT分析
    run_prt_analysis(cfg, data_utils)
    # 运行SRET分析
    run_sret_analysis(cfg, data_utils)


if __name__ == "__main__":
    main()
