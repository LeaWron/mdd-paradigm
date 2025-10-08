import pyxdf
from pathlib import Path


def parse_xdf():
    xdf_path = Path(input("输入 xdf 文件路径:\n").strip('\"'))

    streams, fileheader = pyxdf.load_xdf(xdf_path)

    print(streams)
    print("#" + "=" * 20 + "#")
    print(fileheader)


if __name__ == "__main__":
    parse_xdf()
