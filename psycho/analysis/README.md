# Analysis module

数据分析模块, 这里说明一下大概的结构

---

## entry

整个模块的入口, 调用各个范式实现的地方, 做了一定的交互处理

```python
uv run -m psycho.analysis.entry
```

随后会要求输入日期, session_id, 这里有分支

- 输入日期的话就是处理日期文件夹下的内容
  - 输入格式见 utils.py 中的 parse_date_input
  - 大体上可以直接输入日, 月/日, 年/月/日
  - 不输入直接回车则默认为今天
- 输入 session_id 的话就会和日期一起定位到对应的受试文件,进行单人分析
- 不输入 session_id 则会进入群组分析选择
  - 最多选两个, 每输入一个数字都要回车确认
  - 如果不输入,则会处理日期下的所有受试,但是都是单人分析
  - 最后一定是一个空输入的回车提交
- 群组分析分为单组和两组,取决于上一步选了几个
  - 但是其实单组会同时进行该组别下的所有受试的单人分析
  - 两组也会进行两个组的单组分析,进而也进行所有的单人分析

---

## 范式模块

每个范式也有大致的结构

### 入口

run_`<name>`_analysis 就是这个范式模块暴露出的 api, 也是其入口

run_single... 是单人分析的入口

run_group 是单组分析的入口

run_groups 是多组分析的入口

analyze_`<name>`_data, 真正的调用各个函数的入口

### 共同结构

find_`<name>`_files, 这个是用来寻找给定目录下属于该范式的 csv 文件的函数

load_and_preprocess_data, 加载和预处理数据

- 但是其实没有加载这一步,只有预处理,比如一些新字段的添加,缺少字段的处理,反应时的剪切等

calculate_key_metrics, 这个函数需要返回包含所有 key_metrics 中键的 dict

- 可以是汇总其他函数的计算结果,也可以在这个函数中直接计算, 反正只要结果

create_single_visualizations, 单人分析的可视化创建函数, 按需编写

save_results, 保存你分析的所有结果

create_single_group_visualizations, 单组分析可视化创建

create_multi_group_visualizations, 多组分析可视化

### others

会有一些其他的函数,用来计算中间结果等,按需编写

---

## 工作流

### 单人

run_single_...., 传入文件路径和结果保存目录, 读取文件到 pl.DataFrame, 调用 analyze_..._data

analyze_..._data

- 调用 load_and_preprocess_data 预处理raw DataFrame
- 调用一些其他的函数计算中结果或者进行其他的处理
- 调用 calculate_key_metrics 获得关键指标的 dict
- 调用 create_single_visualizations 创建可视化
- 调用 save_results 保存所有(想保存)的分析结果
- 返回一个dict
  - 这个dict一般需要包含你 save_results 中保存的所有内容(dict)
  - 同时一般也会包含数据的总结,即总共的试次数等等信息

### 单组

run_group, 传入属于该组的文件路径的 list, 结果保存目录, 该组的名称, 参考的组的名称(如果有该项)

- 遍历所有文件, 依次进行单人分析(调用analyze_...),将结果存在 list 中(all_metrics, group_metrics)
  - group_metrics 只保存返回结果中的 key_metrics 项
- 对所有 key_metrics 进行单样本 t 检验,获得效应量
- 通过效应量计算样本量
- 以上两步的结果保存为 statistical_results[metric]
  - metric 为对应的 key_metrics 中的键
- 计算all_metrics 的均值方差
- 统计样本量和效应量的 list[dict]{sample_size_data}
- 调用 create_single_group_visualizations 绘制可视化,得到一个 fig 对象
- 调用 create_common_single_group_figures(见 utils.py), 得到一个  list[go.Figure]
- 调用 save_html_report将以上两者保存为一个 html 可视化文件
- 返回结果
  - 起码包含 all_results, group_metrics, statisitcal_results

注意, 中间会有一些分析结果的保存

### 多组

run_groups, 分别传入对照组和实验组的文件路径的 list, 结果保存目录, 以及两个组对应的名称

- 默认(强制要求)第一个是对照组,第二个是实验组,不过这个其实在 entry 中会进行简单的处理

对两个组分别进行 run_group, 得到结果

检测其正态性和方差齐性(utils中的函数)

调用 perform_group_comparisons(utils) 执行组间比较分析

- 会对所有 key_metrics 计算 t 检验或者 anova 分析
- 返回一个包含效应量,样本量等等的results dict

调用 create_multi_group_visualizations 绘制可视化,得到一个 fig 对象

调用 create_common_comparisons_figures(见 utils.py), 得到一个  list[go.Figure]

调用 save_html_report将以上两者保存为一个 html 可视化文件

返回结果

- 暂时可以不用返回,因为没地方使用到

注意, 中间还有一些分析结果的保存
