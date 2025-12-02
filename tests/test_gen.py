import copy
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import yaml


def check_iat(
    key_seq: list[str],
    max_seq_same: int = 5,
    mean: int = 1,
):
    pre = None
    cnt = 0
    counter = defaultdict(int)
    for key in key_seq:
        if pre == key:
            cnt += 1
            if cnt > max_seq_same:
                return False
        else:
            cnt = 1
        counter[str(key)] += 1
        pre = key
    for key, cnt in counter.items():
        if cnt != mean:
            return False
    return True


@pytest.mark.skip(reason="已生成")
def test_generate_iat(
    stim_path: str | Path = "../psycho/conf/exps/iat/stims.yaml",
    config_path: str | Path = "../psycho/conf/exps/iat/full.yaml",
    save_path: str | Path = "./iat_sequence",
    max_seq_same: int = 5,
):
    base_path = Path(__file__)
    with open(base_path.parent / stim_path, encoding="utf-8") as f:
        stim_dict: dict[str, list[str]] = yaml.safe_load(f)

    with open(base_path.parent / config_path, encoding="utf-8") as f:
        config_dict: dict = yaml.safe_load(f)

    blocks_info: list[dict] = config_dict["blocks_info"]

    sequence = defaultdict(list)
    for block_index, block_config in enumerate(blocks_info):
        n_trials: int = block_config["n_trials"]
        left_kinds: list[str] = block_config["left_kinds"]
        right_kinds: list[str] = block_config["right_kinds"]
        all_kinds = left_kinds + right_kinds

        mean = n_trials // len(all_kinds)
        while True:
            key_seq = np.random.choice(all_kinds, size=n_trials, replace=True)
            if check_iat(key_seq, min(max_seq_same, max(1, mean - 1)), mean):
                break
        seq = []
        for key in key_seq:
            seq.append(random.choice(stim_dict[key]))

        sequence[block_index] = seq

    sequence = dict(sequence)
    with open(Path(save_path).with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml.dump(sequence, f, allow_unicode=True, indent=2)


def check_prt_seq(
    stim_seq: list[str],
    max_seq_same: int = 3,
):
    pre = None
    cnt = 0
    for stim in stim_seq:
        if pre == stim:
            cnt += 1
            if cnt > max_seq_same:
                return False
        else:
            cnt = 1
        pre = stim
    return True


@pytest.mark.skip(reason="已生成")
def test_generate_prt(
    n_blocks: int = 3,
    n_trials_per_block: int = 90,
    max_seq_same: int = 3,
    max_reward_count: int = 40,
    high_low_ratio: float = 3.0,
    seq_save_path: str | Path = "./temp_prt_sequence",
    idx_save_path: str | Path = "./temp_prt_idx_sequence",
):
    sequence = defaultdict(list)
    idx_sequence = {}
    for block_index in range(n_blocks):
        while True:
            half = n_trials_per_block // 2
            stim_seq = ["high"] * half + ["low"] * half
            random.shuffle(stim_seq)

            if check_prt_seq(stim_seq, max_seq_same):
                break

        high_count = int(max_reward_count * (high_low_ratio / (high_low_ratio + 1)))
        low_count = max_reward_count - high_count

        available_high_indices = []
        available_low_indices = []
        for i, stim in enumerate(stim_seq):
            if stim == "high":
                available_high_indices.append(i)
            else:
                available_low_indices.append(i)

        high_indices = sorted(
            np.random.choice(
                list(available_high_indices), size=high_count, replace=False
            ).tolist()
        )

        low_indices = sorted(
            np.random.choice(
                list(available_low_indices), size=low_count, replace=False
            ).tolist()
        )
        sequence[block_index] = stim_seq

        idx_sequence[block_index] = {}
        idx_sequence[block_index]["high"] = high_indices
        idx_sequence[block_index]["low"] = low_indices

    sequence = dict(sequence)
    idx_sequence = dict(idx_sequence)
    with open(Path(idx_save_path).with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml.dump(idx_sequence, f, allow_unicode=True, indent=2)

    with open(Path(seq_save_path).with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml.dump(sequence, f, allow_unicode=True, indent=2)


def check_emotion_face_seq(
    seq: list[dict],
    max_seq_same: int = 1,
):
    pre = None
    cnt = 0
    for item in seq:
        if pre == item["stim_path"]:
            cnt += 1
            if cnt > max_seq_same:
                return False
        else:
            cnt = 1
        pre = item["stim_path"]
    return True


@pytest.mark.skip(reason="已生成")
def test_generate_emotion_face(
    n_blocks: int = 2,
    n_trials_per_block: int = 80,
    max_seq_same: int = 1,
    stim_folder: str | Path = "emotion-face",
    seq_save_path: str | Path = "./emotion_face_sequence",
):
    from psycho.utils import into_stim_str, parse_stim_path

    stim_folder = parse_stim_path(stim_folder)
    stim_sub_folder = list(stim_folder.glob("*-*-*"))
    np.random.shuffle(stim_sub_folder)

    sequence = defaultdict(list)

    for block_index in range(n_blocks):
        block_seq = []
        sub_folder = stim_sub_folder.pop()
        stim_item = list(sub_folder.glob("*.bmp"))
        for i in range(9):
            block_seq.append({"stim_path": into_stim_str(stim_item[i]), "label": 9 - i})
            block_seq.append(
                {"stim_path": into_stim_str(stim_item[-i - 1]), "label": 9 - i}
            )
        block_seq.append({"stim_path": into_stim_str(stim_item[10]), "label": 0})
        block_seq.append({"stim_path": into_stim_str(stim_item[9]), "label": 0})

        np.random.shuffle(block_seq)

        one_group = len(block_seq)
        for _ in range(n_trials_per_block // one_group - 1):
            block_seq.extend(copy.deepcopy(block_seq[:one_group]))

        while True:
            np.random.shuffle(block_seq)
            if check_emotion_face_seq(block_seq, max_seq_same):
                break
        print(len(block_seq))
        sequence[block_index].extend(block_seq)

    with open(Path(seq_save_path).with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml.dump(dict(sequence), f, allow_unicode=True, indent=2)


def check_sret_seq_encoding(
    seq: list[str],
    positive_stim: list[str],
    negative_stim: list[str],
    max_seq_same: int = 1,
):
    cnt = 0
    pre = None
    for item in seq:
        if item in positive_stim:
            if pre == "positive":
                cnt += 1
            else:
                cnt = 1
            pre = "positive"
        elif item in negative_stim:
            if pre == "negative":
                cnt += 1
            else:
                cnt = 1
            pre = "negative"
        if cnt > max_seq_same:
            return False
    return True


@pytest.mark.skip(reason="已生成")
def test_generate_sret(
    max_seq_same: int = 2,
    stim_path: str | Path = "../psycho/conf/exps/sret/stims.yaml",
    seq_save_path: str | Path = "./temp_sret_sequence",
):
    with open(Path(__file__).resolve().parent / stim_path, encoding="utf-8") as f:
        stim_list: dict[str, list[str]] = yaml.safe_load(f)

    sequence = defaultdict(list)

    # encoding phase
    phase = "encoding"
    positive_stim = stim_list["positive"]
    negative_stim = stim_list["negative"]

    candidate = positive_stim + negative_stim
    print("try generate encoding sequence")
    count = 0
    while True:
        np.random.shuffle(candidate)
        if check_sret_seq_encoding(
            candidate,
            positive_stim,
            negative_stim,
            max_seq_same,
        ):
            break
        count += 1
        if count % 100000 == 0:
            print(f"try {count} times")

    sequence[phase] = candidate

    with open(Path(seq_save_path).with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml.dump(dict(sequence), f, allow_unicode=True, indent=2)
