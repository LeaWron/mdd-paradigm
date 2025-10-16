import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import yaml


def check_gng(
    sequence: dict,
    max_seq_same: int = 6,
):
    for block_index, block in sequence.items():
        pre = None
        cnt = 0

        go_cnt, nogo_cnt = 0, 0

        for trial_index, trial in enumerate(block):
            if trial is True:
                go_cnt += 1
            else:
                nogo_cnt += 1
            if pre == trial:
                cnt += 1
                if cnt > max_seq_same:
                    return False
            else:
                cnt = 1
            pre = trial
        if go_cnt != 42 or nogo_cnt != 18:
            return False
    return True


@pytest.mark.skip(reason="已生成")
def test_generate_gng(
    n_blocks: int = 3,
    n_trials_per_block: int = 60,
    max_seq_same: int = 10,
    stim_list: list = [True, False],
    stim_weights: list = [0.7, 0.3],
    save_path: str | Path = "./gng_sequence",
):
    sequence = defaultdict(list)

    stim_nums = [int(n_trials_per_block * w) for w in stim_weights]

    for block_index in range(n_blocks):
        while True:
            candidate = []

            for i in range(len(stim_list)):
                candidate.extend([stim_list[i]] * stim_nums[i])
            random.shuffle(candidate)
            if check_gng(sequence, max_seq_same):
                break

        sequence[block_index] = candidate

    sequence = dict(sequence)
    with open(Path(save_path).with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml.dump(sequence, f, indent=2)


def check_nback(
    target_indices: list,
    n_back: int = 2,
    max_seq_target: int = 3,
):
    target_indices.sort()
    cnt = 0
    for i in range(1, len(target_indices)):
        if target_indices[i] == target_indices[i - 1] + n_back:
            cnt += 1
        else:
            cnt = 0
        if cnt > max_seq_target:
            return False
    return True


@pytest.mark.skip(reason="已生成")
def test_generate_nback(
    n_back: int = 2,
    n_blocks: int = 3,
    n_trials_per_block: int = 60,
    target_ratio: float = 1.0 / 3,
    max_seq_target: int = 4,
    stim_list: list = list(range(1, 10)),
    save_path: str | Path = "./nback_sequence",
):
    sequence = defaultdict(list)
    target_count = int(n_trials_per_block * target_ratio)
    valid_range = list(range(n_back, n_trials_per_block))

    for block_index in range(n_blocks):
        candidate = []
        while True:
            target_indices = np.random.choice(valid_range, size=target_count, replace=False).tolist()
            if check_nback(target_indices, n_back, max_seq_target):
                break

        print(target_indices)
        target_indices = set(target_indices)
        for i in range(n_trials_per_block):
            if i in target_indices:
                candidate.append(candidate[i - n_back])
            else:
                while True:
                    rand = random.choice(stim_list)
                    if len(candidate) < n_back or rand != candidate[-n_back]:
                        break

                candidate.append(rand)

        sequence[block_index] = candidate

    sequence = dict(sequence)
    with open(Path(save_path).with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml.dump(sequence, f, indent=2)


def check_diat(
    key_seq: list[str],
    max_seq_same: int = 5,
    mean: int = 1,
):
    pre = None
    cnt = 0
    for key in key_seq:
        if pre == key:
            cnt += 1
            if cnt > max_seq_same:
                return False
        else:
            cnt = 1
        pre = key
    return True


@pytest.mark.skip(reason="已生成")
def test_generate_diat(
    stim_path: str | Path = "../psycho/conf/exps/diat/stims.yaml",
    config_path: str | Path = "../psycho/conf/exps/diat/full.yaml",
    save_path: str | Path = "./diat_sequence",
    max_seq_same: int = 5,
):
    base_path = Path(__file__)
    with open(base_path.parent / stim_path, "r", encoding="utf-8") as f:
        stim_dict: dict[str, list[str]] = yaml.safe_load(f)

    with open(base_path.parent / config_path, "r", encoding="utf-8") as f:
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
            if check_diat(key_seq, min(max_seq_same, n_trials // 2), mean):
                break
        seq = []
        for key in key_seq:
            seq.append(random.choice(stim_dict[key]))

        sequence[block_index] = seq

    sequence = dict(sequence)
    with open(Path(save_path).with_suffix(".yaml"), "w", encoding="utf-8") as f:
        yaml.dump(sequence, f, allow_unicode=True, indent=2)
