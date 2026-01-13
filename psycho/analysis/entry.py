from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from psycho.analysis.emotion_face import run_emotion_face_analysis
from psycho.analysis.prt import run_prt_analysis
from psycho.analysis.sret import run_sret_analysis
from psycho.analysis.utils import DataUtils, parse_date_input


def run_analysis(
    cfg: DictConfig, date: str = None, session_id: int = None, groups: list[str] = None
):
    data_utils = DataUtils(
        session_id=session_id, date=date, groups=groups, valid_id=cfg.session.valid_id
    )
    # 运行情感分析
    run_emotion_face_analysis(cfg, data_utils)
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

    session_id = input("请入session ID (缺省则进入群组分析模式): \n")

    if session_id:
        session_id = int(session_id)
        run_analysis(cfg, date, session_id)
    else:
        session_id = None
        group_map = {
            "1": "HC",
            "2": "MDD",
            "3": "OHC",
            "4": "OMDD",
        }
        desc = {
            "HC": "对照组",
            "MDD": "MDD组",
            "OHC": "预实验时的对照组",
            "OMDD": "预实验时的MDD组",
        }
        print(
            "请选择要分析的分组(1-4) (可单选或双选, 不选则使用 date 下的所有 session):"
        )
        print(f"1. {group_map['1']} ({desc['HC']})")
        print(f"2. {group_map['2']} ({desc['MDD']})")
        print(f"3. {group_map['3']} ({desc['OHC']})")
        print(f"4. {group_map['4']} ({desc['OMDD']})")

        groups: set[str] = set()

        while True:
            try:
                group = input()
                valid_groups = set(group_map.keys())
                if group not in valid_groups:
                    break
                groups.add(group)
            except ValueError as _:
                break

        match len(groups):
            case 0:
                print("未选择任何分组, 将使用 date 下的所有 session")
                session_set = set()
                for s in (Path(cfg.output_dir) / date).iterdir():
                    if s.is_file() and s.suffix == ".log":
                        session_set.add(int(s.stem.split("-")[0]))

                for session_id in session_set:
                    run_analysis(cfg, date, session_id)
            case 1 | 2:
                # 手动排下序, 0 为 HC, 1 为 MDD
                # HC(1) > OHC(3) > MDD(2) > OMDD(4)
                priority_order = {"1": 0, "3": 1, "2": 2, "4": 3}
                sorted_groups = sorted(groups, key=lambda x: priority_order[x])
                group_list = [group_map[g].lower() for g in sorted_groups]
                print(f"将分析 ({[(g, desc[g.upper()]) for g in group_list]})")

                run_analysis(cfg, date, session_id, groups=group_list)
            case _:
                print("最多只能选择两个分组")
                return


if __name__ == "__main__":
    main()
