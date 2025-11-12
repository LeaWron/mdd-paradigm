from pathlib import Path

import numpy as np
import pyxdf


def parse_xdf(max_samples: int | None = None):
    xdf_path = Path(input("输入 xdf 文件路径:\n").strip('"'))
    print(f"文件: {xdf_path}")
    streams, header = pyxdf.load_xdf(xdf_path)
    print(f"包含 {len(streams)} 个数据流\n")

    for i, stream in enumerate(streams):
        info = stream["info"]
        name = info.get("name", ["<unknown>"])[0]
        stype = info.get("type", ["<unknown>"])[0]
        n_channels = int(info.get("channel_count", ["?"])[0])
        nominal_srate = info.get("nominal_srate", ["?"])[0]

        data = stream["time_series"]
        timestamps = stream["time_stamps"]

        n_samples = len(timestamps)
        print(
            f"[{i}] {name} (type={stype}, channel_count={n_channels}, 采样率={nominal_srate} Hz)"
        )
        print(f"    样本数: {n_samples}")

        # 计算要打印的样本数
        if max_samples is None:
            n_show = n_samples
        else:
            n_show = min(max_samples, n_samples)

        # 标记流 (Marker/Event)
        if stype.lower() in ["markers", "marker", "event"]:
            print(f"    事件数: {len(timestamps)}")
            for j in range(n_show):
                value = (
                    data[j][0] if isinstance(data[j], (list, np.ndarray)) else data[j]
                )
                print(f"      t={timestamps[j]:.3f}s, value={value!r}")
        else:
            # 连续流（EEG, ACC, 等）
            print(f"    前{n_show}个样本:")
            for j in range(n_show):
                d = np.array(data[j])
                short_data = np.array2string(
                    d, precision=3, separator=", ", threshold=10
                )
                print(f"      t={timestamps[j]:.4f}s, data={short_data}")
        print()


if __name__ == "__main__":
    parse_xdf()
