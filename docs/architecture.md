# Architecture

- data_process.py:
  - 使用 `argoverse` 的 api 处理数据，将数据 padding 并转换为 features.
- layers.py:
  - 实现 subgraphlayer 和 self-attention.
- model.py:
  - 实现 subgraph, globalgraph, decoder.
- run.py:
  - 模型训练与评测。
- ./save:
  - 存放处理好的 features 和模型。
- ./docs:
  - 一些文档。