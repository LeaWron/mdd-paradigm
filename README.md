# 范式代码实现

## 项目结构

```plaintext
psycho/                 # 项目根目录
├── pyproject.toml      # （可选，现代项目配置，推荐）
├── README.md
├── psycho/             # 包的真正代码
│   ├── __init__.py     # 让 psycho 变成一个包
│   ├── utils.py        # 工具模块
│   ├── lsl_recv.py     # 接收 lsl 流，通过多进程同步
│   ├── session.py      # session 管理
│   ├── main.py	        # 入口程序
│   └── exps/           # 子包, 所有范式都在这里面实现
│       ├── __init__.py
│       └── gng.py
├── libs/               # 本地文件依赖
│   └── <local_file>.whl
└── tests/              # 单元测试
    └── test_<paradigm>.py

```

以上是项目的结构

- lsl_recv 未必会使用，而是用 [lab_recorder](https://github.com/labstreaminglayer/App-LabRecorder) 代替接收

## 开发

使用 mamba(conda) + uv 进行管理

- 通过 mamba 管理环境
  - mamba create -n <your_environment> uv python=<python_version>
- 通过 uv 管理项目
  - 进入项目根目录
  - uv add `<python-package>` 添加依赖
    - 如果通过本地文件 (.whl) 添加，则将文件放入 libs 文件夹中，并 `uv add ./libs/<local_file>.whl`
  - uv remove 删除依赖
  - uv format 格式化代码

所有范式都放在 psycho/exps/ 下，一般情况下使用如下框架:

```python
from psycho.utils import init_lsl, send_marker


def pre_block():

def block():

def post_block():

def pre_trial():

def trial():

def post_trial():

# win 和 clock 都是从 session 中传递进来的, 如果想方便使用可以在用这两个参数对本地的全局变量赋值
def entry(win, clock): 

def main():
    entry()

if __name__ == "__main__":
    main()
```

但不强制

### Git Commit

建议使用统一提交方式(commitizen)

- uv add commitizen
  - 也可以 uv tool install commitizen, 这种方式需要更新 PATH
- cz init 后一路 enter 确认
- 之后使用 cz commit 代替 git commit

但是也不强制

## 运行

`uv run -m psycho.main` 进入一个 session

如果想单独运行一个实验，需要通过 ``uv run -m psycho.exps.<paradigm> # 没有.py``

## TODO

使用 `hydra` 来管理配置
