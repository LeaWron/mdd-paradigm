# 范式代码实现

## 项目结构

```plaintext
psycho/                              # 项目根目录
├── pyproject.toml                   # （可选，现代项目配置，推荐）
├── README.md
├── psycho/                          # 包的真正代码
│   ├── __init__.py                  # 让 psycho 变成一个包
│   ├── utils.py                     # 工具模块
│   ├── lsl_recv.py                  # 接收 lsl 流，通过多进程同步
│   ├── session.py                   # session 管理
│   ├── main.py	                     # 入口程序
│   ├── exps/                        # 子包, 所有范式都在这里面实现
│   │   ├── __init__.py
│   │   └── gng.py
│   ├── conf/                        # 配置文件目录
│   │   ├── config.yaml              # 入口
│   │   └── exps/                    # 范式配置文件目录
│   │       └── <exp>                # 范式名
│   │          ├── full.yaml         # 正式实验配置
│   │          ├── pre.yaml          # 预实验配置
│   │          └── <other>           # 其他会用到的配置文件
│   └── stims/                       # 存放要使用的刺激的目录
│       └── image_stims.png 
├── libs/                            # 本地文件依赖
│   └── <local_file>.whl
└── tests/                           # 单元测试
    └── test_<paradigm>.py

```

以上是项目的结构

- lsl_recv 未必会使用，而是用 [lab_recorder](https://github.com/labstreaminglayer/App-LabRecorder) 代替接收

## 开发

使用 mamba(conda) + uv 进行管理

- 如果已经在全局有 uv, 则可以忽略 mamba，直接通过 uv 管理项目
- 通过 mamba 管理环境
  - mamba create -n <your_environment> uv python=<python_version>
- 通过 uv 管理项目
  - 进入项目根目录
  - uv add `<python-package>` 添加依赖
    - 如果通过本地文件 (.whl) 添加，则将文件放入 libs 文件夹中，并 `uv add ./libs/<local_file>.whl`
  - uv remove 删除依赖
  - uv format 格式化代码
- 项目中会出现 .venv 等隐藏目录，这是正常的

所有范式都放在 psycho/exps/ 下，一般情况下使用如下框架:

```python
from psycho.utils import init_lsl, send_marker

def pre_block():

def block():

def post_block():

def pre_trial():

def trial():

def post_trial():

# 初始化范式的一些参数, 比如 block 数等全局变量, 根据传入的 cfg 来赋值
# 这个 cfg 会根据预实验与否发生变化
def init_exp(cfg):

# 运行实验, 会在这里调用 init_exp
def run_exp(cfg):


# win 和 clock 都是从 session 中传递进来的, 如果想方便使用可以在用这两个参数对本地的全局变量赋值
# lsl_outlet_session 是 lsl 输出流, cfg 是 hydra 关于这个范式的配置
# 这些参数都可以为 None, 意味着不是由 session 管理, 而是单独运行
def entry(win_session, clock_session, lsl_outlet_session, cfg): 

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

随机生成序列并固定

将 反应时 之类的数据保存为 csv
