from pylsl import StreamInlet, resolve_streams

# from pylsl.resolve import resolve_stream


def main():
    streams = resolve_streams()

    # 连接到第一个符合条件的流
    inlet = StreamInlet(streams[0])

    print("连接成功，开始接收数据：")
    while True:
        sample, timestamp = inlet.pull_sample(timeout=60.0)
        if timestamp is not None and sample is not None:
            print(f"时间戳: {timestamp:.3f}, 数据: {sample[0]}")
        else:
            break


if __name__ == "__main__":
    main()
